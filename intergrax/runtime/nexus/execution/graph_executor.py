# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.long_running.checkpoint_builder import apply_runtime_checkpoint_to_graph
from intergrax.runtime.long_running.runtime_checkpoint import (
    attach_runtime_checkpoint_to_metadata,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryRecord
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task

ExecuteFn = Callable[[Agent, Task, ExecutionNode], Awaitable[AgentExecutionResult]]
ValidateFn = Callable[[AgentExecutionResult, Agent, ExecutionNode], ValidationResult]


class GraphExecutor:
    """
    Executes an ExecutionGraph sequentially by batch, parallel within batch (§25, Phase C.2–C.3).
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        engine: Optional[AgentEngine] = None,
        router: Optional[AgentRouter] = None,
        validation_engine: Optional[NexusValidationEngine] = None,
        retry_engine: Optional[RetryEngine] = None,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self._registry = registry
        self._engine = engine or AgentEngine(registry)
        self._router = router or AgentRouter(registry)
        self._validation_engine = validation_engine or NexusValidationEngine()
        self._retry_engine = retry_engine or RetryEngine(registry)
        self._context_manager = context_manager or ContextManager()

    async def execute(
        self,
        graph: ExecutionGraph,
        task: Task,
        *,
        plan_criteria: Optional[List[str]] = None,
        on_retry: Optional[Callable[[RetryRecord], None]] = None,
        on_node_start: Optional[Callable[[ExecutionNode], None]] = None,
        on_node_complete: Optional[Callable[[ExecutionNode], None]] = None,
    ) -> tuple[List[AgentExecutionResult], List[RetryRecord], ExecutionGraph]:
        prior_outputs: Dict[str, AgentExecutionResult] = {}
        all_executions: List[AgentExecutionResult] = []
        all_retries: List[RetryRecord] = []

        runtime_ckpt = runtime_checkpoint_from_metadata(task.metadata)
        if runtime_ckpt is not None:
            apply_runtime_checkpoint_to_graph(graph, runtime_ckpt, prior_outputs)

        for batch in graph.batches():
            if len(batch) == 1:
                node = batch[0]
                execution, retries, failed = await self._execute_node(
                    graph,
                    task,
                    node,
                    prior_outputs,
                    plan_criteria=plan_criteria,
                    on_retry=on_retry,
                    on_node_start=on_node_start,
                    on_node_complete=on_node_complete,
                )
                all_retries.extend(retries)
                if failed:
                    return all_executions, all_retries, graph
                all_executions.append(execution)
                prior_outputs[node.node_id] = execution
            else:
                results = await asyncio.gather(
                    *[
                        self._execute_node(
                            graph,
                            task,
                            node,
                            prior_outputs,
                            plan_criteria=plan_criteria,
                            on_retry=on_retry,
                            on_node_start=on_node_start,
                            on_node_complete=on_node_complete,
                        )
                        for node in batch
                    ]
                )
                for execution, retries, failed in results:
                    all_retries.extend(retries)
                    if failed:
                        return all_executions, all_retries, graph
                for node in batch:
                    if node.execution_result is not None:
                        all_executions.append(node.execution_result)
                        prior_outputs[node.node_id] = node.execution_result

        return all_executions, all_retries, graph

    async def _execute_node(
        self,
        graph: ExecutionGraph,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        plan_criteria: Optional[List[str]],
        on_retry: Optional[Callable[[RetryRecord], None]],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
    ) -> tuple[AgentExecutionResult, List[RetryRecord], bool]:
        node.status = ExecutionNodeStatus.RUNNING
        if on_node_start is not None:
            on_node_start(node)

        bundle = self._context_manager.build_agent_context(task, node, prior_outputs)
        node_task = self._context_manager.apply_to_task(task, bundle)
        if node.agent_id:
            node_task = node_task.model_copy(update={"agent_id": node.agent_id})

        if node.capability:
            node_task = node_task.model_copy(
                update={
                    "context": node_task.context.model_copy(update={"capability": node.capability}),
                }
            )
        agent = self._router.route(node_task)
        contract = agent.get_contract()
        node_task = node_task.model_copy(update={"agent_id": contract.id})

        def validate_fn(
            execution: AgentExecutionResult,
            current_agent: Agent,
        ) -> ValidationResult:
            return self._validation_engine.validate(
                execution,
                contract=current_agent.get_contract(),
                capability=node.capability or task.context.capability,
                plan_criteria=plan_criteria,
            )

        async def execute_fn(current_agent: Agent) -> AgentExecutionResult:
            request = node_task.to_runtime_request()
            request.metadata["allowed_tools"] = list(current_agent.get_contract().allowed_tools)
            request.metadata["graph_node_id"] = node.node_id
            request.metadata["graph_id"] = graph.graph_id
            plan_id = task.runtime.orchestration.plan_id
            if plan_id:
                request.metadata["plan_id"] = plan_id
            runtime_snapshot = runtime_checkpoint_from_metadata(task.metadata)
            if runtime_snapshot is not None:
                attach_runtime_checkpoint_to_metadata(request.metadata, runtime_snapshot)
            if task.options.human.is_resumed or task.metadata.get("human_approved"):
                request.metadata["human_approved"] = True
            return await AgentEngine.run_agent_with_result(
                current_agent,
                request,
                uaep_executor=self._engine.uaep_executor,
            )

        execution, retries, validation = await self._retry_engine.execute_with_retry(
            node_task,
            agent,
            execute_fn,
            validate_fn=validate_fn,
        )
        if on_retry is not None:
            for record in retries:
                on_retry(record)

        node.execution_result = execution
        if execution.status == AgentExecutionStatus.NEEDS_INPUT:
            node.status = ExecutionNodeStatus.PENDING
            node.metadata["governance_pause"] = True
            if on_node_complete is not None:
                on_node_complete(node)
            return execution, retries, False

        if validation.valid:
            node.status = ExecutionNodeStatus.COMPLETED
        else:
            node.status = ExecutionNodeStatus.FAILED

        if on_node_complete is not None:
            on_node_complete(node)

        return execution, retries, not validation.valid
