# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional, Union

from intergrax.agents.agent_contract import Agent
from intergrax.agents.agent_engine import AgentEngine
from intergrax.agents.persistence.checkpoint_wiring import inject_acp_checkpoint_metadata
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.agents.persistence.compensation_queue_wiring import (
    inject_acp_compensation_queue_metadata,
)
from intergrax.agents.persistence.idempotency_store_wiring import (
    inject_acp_idempotency_store_metadata,
)
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.agents.persistence.tool_invoker_wiring import inject_acp_tool_invoker_metadata
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    AttemptId,
    RunId,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    peek_active_effective_delegation,
    peek_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.contracts.delegation_authority import (
    EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY,
    EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY,
    REQUESTED_PERMISSION_SCOPES_METADATA_KEY,
    DelegationAuthorityError,
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
    resolve_root_parent_execution_authority,
    validate_effective_delegation_metadata_assertions,
    validate_execution_authority_metadata_assertions,
)
from intergrax.contracts.agent_handoff import AgentHandoff, resolve_handoff_from_execution
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    should_skip_graph_node,
)
from intergrax.runtime.long_running.runtime_checkpoint import (
    RuntimeCheckpoint,
    attach_runtime_checkpoint_to_metadata,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.orchestration.swarm_policy import (
    SwarmCoordinationError,
    validate_swarm_parallel_batch,
)
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.metadata_keys import HANDOFF_STRUCTURED_OUTPUT_PREFIX
from intergrax.runtime.nexus.handoff.coordinator import HandoffCoordinator
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionGraphCycleError,
    ExecutionNode,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy, RetryRecord
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task_memory.delegation_memory import TaskMemoryMetadataKey

if TYPE_CHECKING:
    from intergrax.runtime.critic.contracts import CriticVerdict
    from intergrax.runtime.critic.critic_wiring import CriticGraphHooks
    from intergrax.runtime.critic.trace import CriticTraceEmitter
    from intergrax.runtime.nexus.config import RuntimeConfig

ExecuteFn = Callable[[Agent, Task, ExecutionNode], Awaitable[AgentExecutionResult]]
ValidateFn = Callable[[AgentExecutionResult, Agent, ExecutionNode], ValidationResult]
RetryCallback = Callable[[RetryRecord], Union[None, Awaitable[None]]]


async def _notify_retry(
    on_retry: Optional[RetryCallback],
    record: RetryRecord,
) -> None:
    if on_retry is None:
        return
    result = on_retry(record)
    if inspect.isawaitable(result):
        await result
HandoffExtra = tuple[str, AgentExecutionResult]


@dataclass(frozen=True, slots=True)
class _GraphNodeChildRequest:
    graph: ExecutionGraph
    task: Task
    node: ExecutionNode
    prior_outputs: Dict[str, AgentExecutionResult]
    runtime_ckpt: Optional[RuntimeCheckpoint]
    plan_criteria: Optional[List[str]]
    on_retry: Optional[RetryCallback]
    on_node_complete: Optional[Callable[[ExecutionNode], None]]
    critic_trace_emitter: Optional["CriticTraceEmitter"]
    root_execution_authority: ParentExecutionAuthority
    evaluator_loop_active: bool
    agent: Agent
    node_task: Task


@dataclass(frozen=True, slots=True)
class _GraphNodeChildResult:
    execution: AgentExecutionResult
    retries: List[RetryRecord]
    failed: bool
    cancelled: bool
    handoff_extras: List[HandoffExtra]


class _GraphNodeChildDelegate:
    __slots__ = ("_executor",)

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def execute(self, request: _GraphNodeChildRequest) -> _GraphNodeChildResult:
        return await self._executor._execute_node_in_child(request)


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
        handoff_coordinator: Optional[HandoffCoordinator] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        middleware: Optional[MiddlewarePipeline] = None,
        max_parallel_nodes: int | None = None,
        max_inflight_nodes: int | None = None,
        max_delegation_depth: int | None = None,
        critic_graph_hooks: Optional["CriticGraphHooks"] = None,
        agent_checkpoint_store: AgentCheckpointStore | None = None,
        compensation_queue_store: CompensationQueueStore | None = None,
        idempotency_store: IdempotencyStore | None = None,
        declarative_tool_invoker: DeclarativeToolInvoker | None = None,
        runtime_config: Optional["RuntimeConfig"] = None,
        execution_identity: ActiveExecutionIdentity | None = None,
    ) -> None:
        del execution_identity
        self._registry = registry
        self._agent_checkpoint_store = agent_checkpoint_store
        self._compensation_queue_store = compensation_queue_store
        self._idempotency_store = idempotency_store
        self._declarative_tool_invoker = declarative_tool_invoker
        self._runtime_config = runtime_config
        self._graph_routing_step = 0
        self._max_parallel_nodes = max_parallel_nodes
        self._max_inflight_nodes = max_inflight_nodes
        self._max_delegation_depth = max_delegation_depth
        self._inflight_semaphore: asyncio.Semaphore | None = None
        self._engine = engine or AgentEngine(registry)
        self._router = router or AgentRouter(registry, event_bus=event_bus)
        self._validation_engine = validation_engine or NexusValidationEngine()
        self._retry_engine = retry_engine or RetryEngine(
            registry,
            middleware=middleware,
        )
        self._context_manager = context_manager or ContextManager(event_bus=event_bus)
        self._handoff = handoff_coordinator or HandoffCoordinator(registry)
        self._event_bus = event_bus
        self._middleware = middleware or MiddlewarePipeline()
        self._critic_graph_hooks = critic_graph_hooks
        self._child_runner = ChildExecutionRunner[
            _GraphNodeChildRequest,
            _GraphNodeChildResult,
        ]()
        self._graph_node_child_delegate = _GraphNodeChildDelegate(self)
        self._agent_executor = AgentExecutor(self._engine)

    @property
    def execution_identity(self) -> ActiveExecutionIdentity:
        return ActiveExecutionIdentity()

    @property
    def execution_attempt_id(self) -> AttemptId | None:
        bound = peek_active_execution_identity()
        return bound[1] if bound is not None else None

    def _require_run_id(self) -> RunId:
        run_id, _ = require_active_execution_identity()
        return run_id

    def _runtime_event_for_task(self, task: Task, **kwargs: object) -> RuntimeEvent:
        run_id, attempt_id = require_active_execution_identity()
        return RuntimeEvent(
            tenant_id=task.tenant_id,
            task_id=task.task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            correlation_id=task.task_id,
            **kwargs,
        )

    def set_retry_policy(self, policy: RetryPolicy) -> None:
        self._retry_engine = RetryEngine(
            self._registry,
            policy=policy,
            middleware=self._middleware,
        )

    async def execute(
        self,
        graph: ExecutionGraph,
        task: Task,
        *,
        plan_criteria: Optional[List[str]] = None,
        on_retry: Optional[RetryCallback] = None,
        on_node_start: Optional[Callable[[ExecutionNode], None]] = None,
        on_node_complete: Optional[Callable[[ExecutionNode], None]] = None,
        critic_trace_emitter: Optional["CriticTraceEmitter"] = None,
    ) -> tuple[List[AgentExecutionResult], List[RetryRecord], ExecutionGraph, bool]:
        prior_outputs: Dict[str, AgentExecutionResult] = {}
        all_executions: List[AgentExecutionResult] = []
        all_retries: List[RetryRecord] = []

        runtime_ckpt = runtime_checkpoint_from_metadata(task.metadata)
        if runtime_ckpt is not None:
            apply_runtime_checkpoint_to_graph(graph, runtime_ckpt, prior_outputs)

        try:
            batches = graph.batches()
        except ExecutionGraphCycleError as exc:
            failed = AgentExecutionResult(
                agent_id="",
                run_id=self._require_run_id(),
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[str(exc)],
            )
            return [failed], [], graph, False

        root_execution_authority = resolve_root_parent_execution_authority(
            task.execution_authority
        )
        authority_assertion_error = validate_execution_authority_metadata_assertions(
            task.metadata,
            root_execution_authority,
        )
        if authority_assertion_error is not None:
            failed = AgentExecutionResult(
                agent_id="",
                run_id=self._require_run_id(),
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[authority_assertion_error],
            )
            return [failed], [], graph, False

        require_active_execution_identity()
        require_active_execution_id()

        authority_token = None
        if peek_active_execution_authority() is None:
            authority_token = bind_active_execution_authority(root_execution_authority)
        try:
            return await self._execute_graph_batches(
                graph,
                task,
                prior_outputs=prior_outputs,
                all_executions=all_executions,
                all_retries=all_retries,
                runtime_ckpt=runtime_ckpt,
                plan_criteria=plan_criteria,
                on_retry=on_retry,
                on_node_start=on_node_start,
                on_node_complete=on_node_complete,
                critic_trace_emitter=critic_trace_emitter,
                root_execution_authority=root_execution_authority,
            )
        finally:
            if authority_token is not None:
                reset_active_execution_authority(authority_token)

    async def _execute_graph_batches(
        self,
        graph: ExecutionGraph,
        task: Task,
        *,
        prior_outputs: Dict[str, AgentExecutionResult],
        all_executions: List[AgentExecutionResult],
        all_retries: List[RetryRecord],
        runtime_ckpt: Optional[RuntimeCheckpoint],
        plan_criteria: Optional[List[str]],
        on_retry: Optional[RetryCallback],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
        critic_trace_emitter: Optional["CriticTraceEmitter"],
        root_execution_authority: ParentExecutionAuthority,
    ) -> tuple[List[AgentExecutionResult], List[RetryRecord], ExecutionGraph, bool]:
        for batch in graph.batches():
            if CancellationCoordinator.is_requested(task.metadata):
                CancellationCoordinator.mark_pending_graph_nodes_cancelled(graph)
                return all_executions, all_retries, graph, True

            if len(batch) == 1:
                node = batch[0]
                execution, retries, failed, cancelled, handoff_extras = await self._execute_node(
                    graph,
                    task,
                    node,
                    prior_outputs,
                    runtime_ckpt=runtime_ckpt,
                    plan_criteria=plan_criteria,
                    on_retry=on_retry,
                    on_node_start=on_node_start,
                    on_node_complete=on_node_complete,
                    critic_trace_emitter=critic_trace_emitter,
                    root_execution_authority=root_execution_authority,
                )
                all_retries.extend(retries)
                if cancelled:
                    CancellationCoordinator.mark_pending_graph_nodes_cancelled(graph)
                    return all_executions, all_retries, graph, True
                if failed:
                    return all_executions, all_retries, graph, False
                all_executions.append(execution)
                prior_outputs[node.node_id] = execution
                for node_id, extra_execution in handoff_extras:
                    all_executions.append(extra_execution)
                    prior_outputs[node_id] = extra_execution
            else:
                coordination_pattern = task.metadata.get("coordination_pattern")
                if coordination_pattern == CoordinationPattern.SWARM.value:
                    try:
                        validate_swarm_parallel_batch(len(batch))
                    except SwarmCoordinationError as exc:
                        failed = AgentExecutionResult(
                            agent_id="",
                            run_id=self._require_run_id(),
                            status=AgentExecutionStatus.FAILED,
                            summary="",
                            errors=[str(exc)],
                        )
                        return [failed], [], graph, False
                results = await self._execute_parallel_batch(
                    batch,
                    graph=graph,
                    task=task,
                    prior_outputs=prior_outputs,
                    runtime_ckpt=runtime_ckpt,
                    plan_criteria=plan_criteria,
                    on_retry=on_retry,
                    on_node_start=on_node_start,
                    on_node_complete=on_node_complete,
                    critic_trace_emitter=critic_trace_emitter,
                    root_execution_authority=root_execution_authority,
                )
                for execution, retries, failed, cancelled, handoff_extras in results:
                    all_retries.extend(retries)
                    if cancelled:
                        CancellationCoordinator.mark_pending_graph_nodes_cancelled(graph)
                        return all_executions, all_retries, graph, True
                    if failed:
                        return all_executions, all_retries, graph, False
                    for node_id, extra_execution in handoff_extras:
                        all_executions.append(extra_execution)
                        prior_outputs[node_id] = extra_execution
                for node in batch:
                    if node.execution_result is not None:
                        all_executions.append(node.execution_result)
                        prior_outputs[node.node_id] = node.execution_result

        return all_executions, all_retries, graph, False

    async def _execute_parallel_batch(
        self,
        batch: list[ExecutionNode],
        *,
        graph: ExecutionGraph,
        task: Task,
        prior_outputs: Dict[str, AgentExecutionResult],
        runtime_ckpt: Optional[RuntimeCheckpoint],
        plan_criteria: Optional[List[str]],
        on_retry: Optional[RetryCallback],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
        critic_trace_emitter: Optional["CriticTraceEmitter"] = None,
        root_execution_authority: ParentExecutionAuthority,
    ) -> list[
        tuple[
            AgentExecutionResult,
            List[RetryRecord],
            bool,
            bool,
            List[HandoffExtra],
        ]
    ]:
        if self._inflight_semaphore is None and self._max_inflight_nodes is not None:
            self._inflight_semaphore = asyncio.Semaphore(self._max_inflight_nodes)

        limit = self._max_parallel_nodes
        if limit is None or limit >= len(batch):
            return list(
                await asyncio.gather(
                    *[
                        self._execute_node(
                            graph,
                            task,
                            node,
                            prior_outputs,
                            runtime_ckpt=runtime_ckpt,
                            plan_criteria=plan_criteria,
                            on_retry=on_retry,
                            on_node_start=on_node_start,
                            on_node_complete=on_node_complete,
                            critic_trace_emitter=critic_trace_emitter,
                            root_execution_authority=root_execution_authority,
                        )
                        for node in batch
                    ]
                )
            )

        semaphore = asyncio.Semaphore(limit)

        async def _run_node(
            node: ExecutionNode,
        ) -> tuple[
            AgentExecutionResult,
            List[RetryRecord],
            bool,
            bool,
            List[HandoffExtra],
        ]:
            inflight = self._inflight_semaphore
            if inflight is not None and inflight.locked():
                await self._emit_backpressure(task, node.node_id)
            async with semaphore:
                if inflight is not None:
                    async with inflight:
                        return await self._execute_node(
                            graph,
                            task,
                            node,
                            prior_outputs,
                            runtime_ckpt=runtime_ckpt,
                            plan_criteria=plan_criteria,
                            on_retry=on_retry,
                            on_node_start=on_node_start,
                            on_node_complete=on_node_complete,
                            critic_trace_emitter=critic_trace_emitter,
                            root_execution_authority=root_execution_authority,
                        )
                return await self._execute_node(
                    graph,
                    task,
                    node,
                    prior_outputs,
                    runtime_ckpt=runtime_ckpt,
                    plan_criteria=plan_criteria,
                    on_retry=on_retry,
                    on_node_start=on_node_start,
                    on_node_complete=on_node_complete,
                    critic_trace_emitter=critic_trace_emitter,
                    root_execution_authority=root_execution_authority,
                )

        return list(await asyncio.gather(*[_run_node(node) for node in batch]))

    async def _execute_node(
        self,
        graph: ExecutionGraph,
        task: Task,
        node: ExecutionNode,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        runtime_ckpt: Optional[RuntimeCheckpoint] = None,
        plan_criteria: Optional[List[str]],
        on_retry: Optional[Callable[[RetryRecord], None]],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
        critic_trace_emitter: Optional["CriticTraceEmitter"] = None,
        root_execution_authority: ParentExecutionAuthority,
        evaluator_loop_active: bool = False,
    ) -> tuple[AgentExecutionResult, List[RetryRecord], bool, bool, List[HandoffExtra]]:
        if should_skip_graph_node(
            node,
            checkpoint=runtime_ckpt,
            prior_outputs=prior_outputs,
        ):
            execution = prior_outputs[node.node_id]
            node.execution_result = execution
            node.status = ExecutionNodeStatus.SKIPPED
            if on_node_complete is not None:
                on_node_complete(node)
            return execution, [], False, False, []

        if CancellationCoordinator.is_requested(task.metadata):
            node.status = ExecutionNodeStatus.SKIPPED
            node.metadata["cancelled"] = True
            if on_node_complete is not None:
                on_node_complete(node)
            return (
                AgentExecutionResult(
                    agent_id=node.agent_id or "",
                    run_id=self._require_run_id(),
                    status=AgentExecutionStatus.FAILED,
                    summary="",
                    errors=["task_cancelled"],
                ),
                [],
                False,
                True,
                [],
            )

        node.status = ExecutionNodeStatus.RUNNING
        if on_node_start is not None:
            on_node_start(node)

        from intergrax.runtime.nexus.context.routing_snapshot_sync import sync_routing_for_graph_task

        sync_routing_for_graph_task(
            task,
            step_index=self._graph_routing_step,
            runtime_config=self._runtime_config,
        )
        self._graph_routing_step += 1

        delegation_error = self._validate_delegation_constraints(
            graph,
            node,
            root_execution_authority=root_execution_authority,
        )
        if delegation_error is not None:
            failed = AgentExecutionResult(
                agent_id=node.agent_id or "",
                run_id=self._require_run_id(),
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[delegation_error],
            )
            node.execution_result = failed
            node.status = ExecutionNodeStatus.FAILED
            if on_node_complete is not None:
                on_node_complete(node)
            return failed, [], True, False, []

        bundle = await self._context_manager.build_agent_context_async(
            task, node, prior_outputs
        )
        node_task = self._context_manager.apply_to_task(task, bundle)
        if node.agent_id:
            node_task = node_task.model_copy(update={"agent_id": node.agent_id})

        if node.capability:
            node_task = node_task.model_copy(
                update={
                    "context": node_task.context.model_copy(update={"capability": node.capability}),
                }
            )

        selection_ctx = HookContext(
            task_id=task.task_id,
            run_id=self._require_run_id(),
            node_id=node.node_id,
            agent_id=node.agent_id,
            phase=ExecutionPhase.AGENT_SELECTION,
            runtime_state={"capability": node.capability or task.context.capability},
        )
        before_selection = await self._middleware.run_before(
            HookPoint.BEFORE_AGENT_SELECTION,
            selection_ctx,
        )
        if before_selection.action != HookAction.ALLOW:
            failed = AgentExecutionResult(
                agent_id=node.agent_id or "",
                run_id=self._require_run_id(),
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[before_selection.reason or "agent_selection_blocked_by_hook"],
            )
            node.execution_result = failed
            node.status = ExecutionNodeStatus.FAILED
            if on_node_complete is not None:
                on_node_complete(node)
            return failed, [], True, False, []

        active_run_id = self._require_run_id()
        agent = self._router.route(
            node_task,
            run_id=active_run_id,
            node_id=node.node_id,
        )
        contract = agent.get_contract()
        node_task = node_task.model_copy(update={"agent_id": contract.id})

        await self._middleware.run_after(
            HookPoint.AFTER_AGENT_SELECTION,
            selection_ctx.model_copy(update={"agent_id": contract.id}),
        )

        child_request = _GraphNodeChildRequest(
            graph=graph,
            task=task,
            node=node,
            prior_outputs=prior_outputs,
            runtime_ckpt=runtime_ckpt,
            plan_criteria=plan_criteria,
            on_retry=on_retry,
            on_node_complete=on_node_complete,
            critic_trace_emitter=critic_trace_emitter,
            root_execution_authority=root_execution_authority,
            evaluator_loop_active=evaluator_loop_active,
            agent=agent,
            node_task=node_task,
        )
        requested_permission_scopes: tuple[str, ...] | None = None
        if node.delegation is not None:
            requested_permission_scopes = node.delegation.permission_scopes
        try:
            child_result = await self._child_runner.execute(
                request=child_request,
                delegate=self._graph_node_child_delegate,
                requested_permission_scopes=requested_permission_scopes,
            )
        except DelegationAuthorityError as exc:
            failed = AgentExecutionResult(
                agent_id=node.agent_id or "",
                run_id=self._require_run_id(),
                status=AgentExecutionStatus.FAILED,
                summary="",
                errors=[str(exc)],
            )
            node.execution_result = failed
            node.status = ExecutionNodeStatus.FAILED
            if on_node_complete is not None:
                on_node_complete(node)
            return failed, [], True, False, []
        return (
            child_result.execution,
            child_result.retries,
            child_result.failed,
            child_result.cancelled,
            child_result.handoff_extras,
        )

    async def _execute_node_in_child(
        self,
        child_request: _GraphNodeChildRequest,
    ) -> _GraphNodeChildResult:
        graph = child_request.graph
        task = child_request.task
        node = child_request.node
        prior_outputs = child_request.prior_outputs
        runtime_ckpt = child_request.runtime_ckpt
        plan_criteria = child_request.plan_criteria
        on_retry = child_request.on_retry
        on_node_complete = child_request.on_node_complete
        critic_trace_emitter = child_request.critic_trace_emitter
        root_execution_authority = child_request.root_execution_authority
        evaluator_loop_active = child_request.evaluator_loop_active
        agent = child_request.agent
        node_task = child_request.node_task

        if node.delegation is not None:
            effective_authority = peek_active_effective_delegation()
            if effective_authority is None:
                raise RuntimeError(
                    "effective delegation evidence required for delegated node execution"
                )
            node.metadata[EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY] = effective_authority
            parent_agent_id = task.agent_id or node.agent_id or agent.get_contract().id
            await self._emit_delegation_granted(
                task,
                node,
                parent_agent_id=parent_agent_id or "",
                child_agent_id=agent.get_contract().id,
            )

        last_critic_verdict: Optional["CriticVerdict"] = None

        def validate_fn(
            execution: AgentExecutionResult,
            current_agent: Agent,
        ) -> ValidationResult:
            nonlocal last_critic_verdict
            contract = current_agent.get_contract()
            cap = node.capability or task.context.capability
            if (
                self._critic_graph_hooks is not None
                and self._critic_graph_hooks.verify_node_partial
            ):
                from intergrax.runtime.critic.critic_wiring import validate_node_with_critic_detail

                validation, verdict = validate_node_with_critic_detail(
                    execution,
                    contract=contract,
                    hooks=self._critic_graph_hooks,
                    task_id=task.task_id,
                    run_id=self._require_run_id(),
                    tenant_id=task.tenant_id,
                    capability=cap,
                    plan_criteria=plan_criteria,
                    trace_emitter=critic_trace_emitter,
                    node_id=node.node_id,
                )
                last_critic_verdict = verdict
                return validation
            last_critic_verdict = None
            return self._validation_engine.validate(
                execution,
                contract=contract,
                capability=cap,
                plan_criteria=plan_criteria,
            )

        async def execute_fn(current_agent: Agent) -> AgentExecutionResult:
            active_run_id = self._require_run_id()
            selected_agent_id = current_agent.get_contract().id
            request = node_task.model_copy(
                update={"agent_id": selected_agent_id},
            ).to_runtime_request(run_id=active_run_id)
            from intergrax.runtime.human.declarative_hitl_grant import (
                DeclarativeHitlGrantCoordinator,
            )

            request = DeclarativeHitlGrantCoordinator.transfer_persisted_grant_for_resume(
                task, request
            )
            inject_acp_checkpoint_metadata(
                request.metadata,
                store=self._agent_checkpoint_store,
                run_id=active_run_id,
                tenant_id=task.tenant_id,
            )
            inject_acp_tool_invoker_metadata(
                request.metadata,
                self._declarative_tool_invoker,
                task_id=task.task_id,
                run_id=active_run_id,
                agent_id=selected_agent_id,
                tenant_id=task.tenant_id,
            )
            inject_acp_compensation_queue_metadata(
                request.metadata,
                self._compensation_queue_store,
            )
            inject_acp_idempotency_store_metadata(
                request.metadata,
                self._idempotency_store,
            )
            from intergrax.runtime.workspace.exec_ctx_isolation import (
                RUNTIME_SANDBOX_MANAGER_METADATA_KEY,
                RUNTIME_SHADOW_MANAGER_METADATA_KEY,
            )

            request.metadata[RUNTIME_SHADOW_MANAGER_METADATA_KEY] = self._engine.shadow_manager
            request.metadata[RUNTIME_SANDBOX_MANAGER_METADATA_KEY] = self._engine.sandbox_manager
            resolved_contract = (
                self._registry.get_contract(selected_agent_id)
                if self._registry.has(selected_agent_id)
                else current_agent.get_contract()
            )
            request.metadata["allowed_tools"] = list(resolved_contract.allowed_tools)
            request.metadata["graph_node_id"] = node.node_id
            request.metadata["graph_id"] = graph.graph_id
            plan_id = task.runtime.orchestration.plan_id
            if plan_id:
                request.metadata["plan_id"] = plan_id
            runtime_snapshot = runtime_checkpoint_from_metadata(task.metadata)
            if runtime_snapshot is not None:
                attach_runtime_checkpoint_to_metadata(request.metadata, runtime_snapshot)
            critic_feedback = node.metadata.get("critic_feedback")
            if isinstance(critic_feedback, list) and critic_feedback:
                request.metadata["critic_feedback"] = list(critic_feedback)
            if node.delegation is not None:
                delegation = node.delegation
                effective_authority = node.metadata.get(EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY)
                request.metadata[TaskMemoryMetadataKey.DELEGATION_MEMORY_NAMESPACE] = (
                    delegation.resolved_memory_namespace(
                        task_id=task.task_id,
                        node_id=node.node_id,
                    )
                )
                if isinstance(effective_authority, EffectiveDelegationAuthority):
                    request.effective_delegation_authority = effective_authority
                    request.metadata[EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY] = list(
                        effective_authority.effective_permission_scopes
                    )
                    request.metadata[REQUESTED_PERMISSION_SCOPES_METADATA_KEY] = list(
                        effective_authority.requested_permission_scopes
                    )
                    metadata_conflict = validate_effective_delegation_metadata_assertions(
                        request.metadata,
                        effective_authority,
                    )
                    if metadata_conflict is not None:
                        return AgentExecutionResult(
                            agent_id=selected_agent_id,
                            run_id=active_run_id,
                            status=AgentExecutionStatus.FAILED,
                            summary="",
                            errors=[metadata_conflict],
                        )
                request.metadata["run_id"] = f"{task.task_id}:{node.node_id}"
                if delegation.parent_run_id:
                    request.metadata[TaskMemoryMetadataKey.PARENT_RUN_ID] = delegation.parent_run_id
                if delegation.parent_node_id:
                    request.metadata[TaskMemoryMetadataKey.PARENT_NODE_ID] = delegation.parent_node_id
                if delegation.explore is not None:
                    from intergrax.runtime.nexus.delegation.explore_integration import (
                        apply_explore_delegation_context,
                    )

                    apply_explore_delegation_context(
                        request,
                        delegation,
                        task_id=task.task_id,
                        node_id=node.node_id,
                    )
            CancellationCoordinator.propagate(task.metadata, request.metadata)
            replan_policy = task.metadata.get("replan_policy.v1")
            if isinstance(replan_policy, dict):
                request.metadata["replan_policy.v1"] = replan_policy
            engine_prompt_id = task.metadata.get("engine_planner_prompt_id")
            if isinstance(engine_prompt_id, str) and engine_prompt_id.strip():
                request.metadata["engine_planner_prompt_id"] = engine_prompt_id.strip()
            execution_request = ExecutionRequest(
                input=request,
                output_type=AgentExecutionResult,
                capabilities=frozenset({ExecutionCapability.AGENT}),
            )
            governed_task_binding = ActiveGovernedExecutionTask()
            token = governed_task_binding.bind(task)
            try:
                execution_result = await self._agent_executor.execute(execution_request)
            finally:
                governed_task_binding.reset(token)
            return execution_result.output

        execution, retries, validation = await self._retry_engine.execute_with_retry(
            node_task,
            agent,
            execute_fn,
            validate_fn=validate_fn,
        )
        if (
            not validation.valid
            and last_critic_verdict is not None
            and self._critic_graph_hooks is not None
            and not evaluator_loop_active
        ):
            execution, validation, retries = await self._maybe_run_evaluator_loop(
                graph,
                task,
                node,
                execution,
                validation,
                last_critic_verdict,
                prior_outputs=prior_outputs,
                runtime_ckpt=runtime_ckpt,
                plan_criteria=plan_criteria,
                on_retry=on_retry,
                on_node_start=None,
                on_node_complete=on_node_complete,
                critic_trace_emitter=critic_trace_emitter,
                agent=agent,
                node_task=node_task,
                execute_fn=execute_fn,
                validate_fn=validate_fn,
                prior_retries=retries,
                root_execution_authority=root_execution_authority,
            )
        if CancellationCoordinator.is_requested(task.metadata):
            node.execution_result = execution
            node.status = ExecutionNodeStatus.SKIPPED
            node.metadata["cancelled"] = True
            if on_node_complete is not None:
                on_node_complete(node)
            return _GraphNodeChildResult(
                execution=execution,
                retries=retries,
                failed=False,
                cancelled=True,
                handoff_extras=[],
            )

        for record in retries:
            await _notify_retry(on_retry, record)

        node.execution_result = execution
        if execution.status == AgentExecutionStatus.NEEDS_INPUT:
            node.status = ExecutionNodeStatus.PENDING
            node.metadata["governance_pause"] = True
            if on_node_complete is not None:
                on_node_complete(node)
            return _GraphNodeChildResult(
                execution=execution,
                retries=retries,
                failed=False,
                cancelled=False,
                handoff_extras=[],
            )

        if validation.valid:
            node.status = ExecutionNodeStatus.COMPLETED
            self._context_manager.record_node_output(task, node, execution)
            handoff_extras = await self._maybe_execute_handoff(
                graph,
                task,
                node,
                execution,
                prior_outputs,
                runtime_ckpt=runtime_ckpt,
                plan_criteria=plan_criteria,
                on_retry=on_retry,
                on_node_start=None,
                on_node_complete=on_node_complete,
                critic_trace_emitter=critic_trace_emitter,
                root_execution_authority=root_execution_authority,
            )
            failed = False
        else:
            node.status = ExecutionNodeStatus.FAILED
            handoff_extras = []
            failed = True

        if on_node_complete is not None:
            on_node_complete(node)

        return _GraphNodeChildResult(
            execution=execution,
            retries=retries,
            failed=failed,
            cancelled=False,
            handoff_extras=handoff_extras,
        )

    async def _maybe_run_evaluator_loop(
        self,
        graph: ExecutionGraph,
        task: Task,
        node: ExecutionNode,
        execution: AgentExecutionResult,
        validation: ValidationResult,
        verdict: "CriticVerdict",
        *,
        prior_outputs: Dict[str, AgentExecutionResult],
        runtime_ckpt: Optional[RuntimeCheckpoint],
        plan_criteria: Optional[List[str]],
        on_retry: Optional[Callable[[RetryRecord], None]],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
        critic_trace_emitter: Optional["CriticTraceEmitter"],
        agent: Agent,
        node_task: Task,
        execute_fn: Callable[[Agent], Awaitable[AgentExecutionResult]],
        validate_fn: ValidateFn,
        prior_retries: List[RetryRecord],
        root_execution_authority: ParentExecutionAuthority,
    ) -> tuple[AgentExecutionResult, ValidationResult, List[RetryRecord]]:
        from intergrax.runtime.critic.critic_wiring import validate_node_with_critic_detail
        from intergrax.runtime.critic.evaluator_loop_executor import (
            EvaluatorLoopDecision,
            EvaluatorLoopExecutor,
            EvaluatorLoopIterationState,
        )
        from intergrax.runtime.critic.evaluator_loop_metadata import (
            current_evaluator_loop_iteration,
            evaluator_loop_spec_from_node,
            set_evaluator_loop_iteration,
        )

        if self._critic_graph_hooks is None:
            return execution, validation, prior_retries

        spec = evaluator_loop_spec_from_node(node)
        if spec is None:
            return execution, validation, prior_retries

        loop_executor = EvaluatorLoopExecutor(
            spec=spec,
            trace_emitter=critic_trace_emitter,
        )
        state = EvaluatorLoopIterationState(
            worker_node_id=node.node_id,
            iteration=current_evaluator_loop_iteration(node),
        )
        current_execution = execution
        current_validation = validation
        current_verdict = verdict
        all_retries = list(prior_retries)
        contract = agent.get_contract()
        cap = node.capability or task.context.capability

        while not current_validation.valid:
            outcome = loop_executor.decide_after_verdict(
                current_verdict,
                state=state,
                tenant_id=task.tenant_id,
                task_id=task.task_id,
                agent_id=contract.id,
                node_id=node.node_id,
            )
            if outcome.decision is EvaluatorLoopDecision.CONTINUE:
                break
            if outcome.decision is EvaluatorLoopDecision.REVISE and outcome.revise_node_id:
                if current_verdict.failure_reasons:
                    node.metadata["critic_feedback"] = list(current_verdict.failure_reasons)
                merged_prior = dict(prior_outputs)
                merged_prior[node.node_id] = current_execution
                revise_node_id = outcome.revise_node_id
                if revise_node_id != node.node_id:
                    revise_node = graph.node_by_id(revise_node_id)
                    await self._execute_node(
                        graph,
                        task,
                        revise_node,
                        merged_prior,
                        runtime_ckpt=runtime_ckpt,
                        plan_criteria=plan_criteria,
                        on_retry=on_retry,
                        on_node_start=on_node_start,
                        on_node_complete=on_node_complete,
                        critic_trace_emitter=critic_trace_emitter,
                        root_execution_authority=root_execution_authority,
                        evaluator_loop_active=True,
                    )
                state = loop_executor.bump_iteration(state)
                set_evaluator_loop_iteration(node, state.iteration)
                current_execution, loop_retries, current_validation = (
                    await self._retry_engine.execute_with_retry(
                        node_task,
                        agent,
                        execute_fn,
                        validate_fn=validate_fn,
                    )
                )
                all_retries.extend(loop_retries)
                if self._critic_graph_hooks.verify_node_partial:
                    current_validation, current_verdict = validate_node_with_critic_detail(
                        current_execution,
                        contract=contract,
                        hooks=self._critic_graph_hooks,
                        task_id=task.task_id,
                        run_id=self._require_run_id(),
                        tenant_id=task.tenant_id,
                        capability=cap,
                        plan_criteria=plan_criteria,
                        trace_emitter=critic_trace_emitter,
                        node_id=node.node_id,
                    )
                continue
            if outcome.decision is EvaluatorLoopDecision.ESCALATE_HITL:
                current_execution = current_execution.model_copy(
                    update={"status": AgentExecutionStatus.NEEDS_INPUT},
                )
                current_validation = ValidationResult(
                    valid=False,
                    errors=list(outcome.failure_reasons) or ["critic_escalate_hitl"],
                )
                break
            break

        return current_execution, current_validation, all_retries

    async def _maybe_execute_handoff(
        self,
        graph: ExecutionGraph,
        task: Task,
        node: ExecutionNode,
        execution: AgentExecutionResult,
        prior_outputs: Dict[str, AgentExecutionResult],
        *,
        runtime_ckpt: Optional[RuntimeCheckpoint],
        plan_criteria: Optional[List[str]],
        on_retry: Optional[Callable[[RetryRecord], None]],
        on_node_start: Optional[Callable[[ExecutionNode], None]],
        on_node_complete: Optional[Callable[[ExecutionNode], None]],
        critic_trace_emitter: Optional["CriticTraceEmitter"] = None,
        root_execution_authority: ParentExecutionAuthority,
    ) -> List[HandoffExtra]:
        handoff = resolve_handoff_from_execution(execution)
        if handoff is None:
            return []

        validation = self._handoff.validate(
            handoff,
            from_agent_id=execution.agent_id,
            task=task,
        )
        if not validation.valid or validation.resolved_agent_id is None:
            node.status = ExecutionNodeStatus.FAILED
            node.metadata["handoff_validation_errors"] = validation.errors
            execution.errors.extend(validation.errors)
            return []

        hook_ctx = HookContext(
            task_id=task.task_id,
            run_id=self._require_run_id(),
            node_id=node.node_id,
            agent_id=execution.agent_id,
            phase=ExecutionPhase.STEP_EXECUTION,
        )
        before = await self._middleware.run_before(HookPoint.BEFORE_HANDOFF, hook_ctx)
        if before.action != HookAction.ALLOW:
            node.status = ExecutionNodeStatus.FAILED
            execution.errors.append(before.reason or "handoff blocked by hook")
            return []

        await self._emit_handoff_event(
            task,
            handoff,
            RuntimeEventType.HANDOFF_INITIATED,
            from_node_id=node.node_id,
            to_agent_id=validation.resolved_agent_id,
        )

        self._context_manager.put_structured_output(
            task,
            key=f"{HANDOFF_STRUCTURED_OUTPUT_PREFIX}{handoff.handoff_id}",
            payload={
                "from_agent_id": handoff.from_agent_id,
                "to_agent_id": validation.resolved_agent_id,
                "reason": handoff.reason,
                "payload": dict(handoff.payload),
                "artifacts": list(handoff.artifacts),
            },
        )

        handoff_node = self._handoff.apply_to_graph(
            graph,
            handoff,
            from_node_id=node.node_id,
            resolved_agent_id=validation.resolved_agent_id,
        )

        merged_prior = dict(prior_outputs)
        merged_prior[node.node_id] = execution
        handoff_execution, handoff_retries, handoff_failed, _, nested_extras = await self._execute_node(
            graph,
            task,
            handoff_node,
            merged_prior,
            runtime_ckpt=runtime_ckpt,
            plan_criteria=plan_criteria,
            on_retry=on_retry,
            on_node_start=on_node_start,
            on_node_complete=on_node_complete,
            critic_trace_emitter=critic_trace_emitter,
            root_execution_authority=root_execution_authority,
        )
        for record in handoff_retries:
            await _notify_retry(on_retry, record)

        if handoff_failed or handoff_execution.status != AgentExecutionStatus.COMPLETED:
            node.metadata["handoff_failed"] = True
            return []

        await self._emit_handoff_event(
            task,
            handoff,
            RuntimeEventType.HANDOFF_COMPLETED,
            from_node_id=node.node_id,
            to_agent_id=validation.resolved_agent_id,
            handoff_node_id=handoff_node.node_id,
        )
        await self._middleware.run_after(
            HookPoint.AFTER_HANDOFF,
            hook_ctx.model_copy(update={"agent_id": validation.resolved_agent_id}),
        )

        extras: List[HandoffExtra] = [(handoff_node.node_id, handoff_execution)]
        extras.extend(nested_extras)
        return extras

    async def _emit_delegation_granted(
        self,
        task: Task,
        node: ExecutionNode,
        *,
        parent_agent_id: str,
        child_agent_id: str,
    ) -> None:
        if self._event_bus is None or node.delegation is None:
            return
        delegation = node.delegation
        effective_authority = node.metadata.get(EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY)
        if not isinstance(effective_authority, EffectiveDelegationAuthority):
            return
        payload: dict[str, object] = {
            "parent_agent_id": parent_agent_id,
            "child_agent_id": child_agent_id,
            "node_id": node.node_id,
            "rationale": delegation.objective,
            "requested_permission_scopes": list(
                effective_authority.requested_permission_scopes
            ),
            "effective_permission_scopes": list(
                effective_authority.effective_permission_scopes
            ),
            "permission_scopes": list(effective_authority.effective_permission_scopes),
        }
        await self._event_bus.publish(
            self._runtime_event_for_task(
                task,
                node_id=node.node_id,
                agent_id=parent_agent_id,
                event_type=RuntimeEventType.DELEGATION_GRANTED,
                phase=ExecutionPhase.STEP_EXECUTION,
                payload=payload,
            )
        )

    async def _emit_handoff_event(
        self,
        task: Task,
        handoff: AgentHandoff,
        event_type: RuntimeEventType,
        *,
        from_node_id: str,
        to_agent_id: str,
        handoff_node_id: Optional[str] = None,
    ) -> None:
        if self._event_bus is None:
            return
        payload: dict[str, object] = {
            "handoff_id": handoff.handoff_id,
            "from_agent_id": handoff.from_agent_id,
            "from_node_id": from_node_id,
            "to_agent_id": to_agent_id,
            "to_capability": handoff.to_capability,
            "reason": handoff.reason,
        }
        if handoff_node_id is not None:
            payload["handoff_node_id"] = handoff_node_id
        await self._event_bus.publish(
            self._runtime_event_for_task(
                task,
                node_id=handoff_node_id or from_node_id,
                agent_id=handoff.from_agent_id,
                event_type=event_type,
                phase=ExecutionPhase.STEP_EXECUTION,
                payload=payload,
            )
        )

    async def _emit_backpressure(self, task: Task, node_id: str) -> None:
        if self._event_bus is None:
            return
        await self._event_bus.publish(
            self._runtime_event_for_task(
                task,
                node_id=node_id,
                event_type=RuntimeEventType.GRAPH_BACKPRESSURE,
                phase=ExecutionPhase.STEP_EXECUTION,
                payload={"max_inflight_nodes": self._max_inflight_nodes},
            )
        )

    def _delegation_depth(self, graph: ExecutionGraph, node: ExecutionNode) -> int:
        if node.delegation is None:
            return 0
        parent_depth = 0
        for dep_id in node.depends_on:
            dep = graph.node_by_id(dep_id)
            parent_depth = max(parent_depth, self._delegation_depth(graph, dep))
        return 1 + parent_depth

    def _validate_delegation_constraints(
        self,
        graph: ExecutionGraph,
        node: ExecutionNode,
        *,
        root_execution_authority: ParentExecutionAuthority | None = None,
    ) -> str | None:
        if node.delegation is None:
            return None
        if self._max_delegation_depth is not None:
            depth = self._delegation_depth(graph, node)
            if depth > self._max_delegation_depth:
                return (
                    f"max_delegation_depth exceeded: depth={depth} "
                    f"limit={self._max_delegation_depth}"
                )
        delegation = node.delegation
        if delegation.max_llm_calls is not None and delegation.max_llm_calls < 1:
            return "delegation_budget_llm_calls_exhausted"
        if delegation.max_tool_calls is not None and delegation.max_tool_calls < 1:
            return "delegation_budget_tool_calls_exhausted"
        return None
