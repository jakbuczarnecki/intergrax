# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical host task root execution (UE-11G-P1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.agentic import AgentEnginePort
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.execution_terminal.persistence import (
    terminal_outcome_from_task_state,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.orchestration import (
    OrchestrationExecutor,
    TaskBoundOrchestrationDelegate,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
    mint_root_execution_identity,
)
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput, execution_request_from_task
from intergrax.runtime.execution.host_task_terminal_publisher import HostTaskTerminalPublisher
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.orchestration_capabilities import is_orchestration_capability
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState


def resolve_task_execution_capabilities(
    task: Task,
    *,
    orchestration_triggers: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
) -> frozenset[ExecutionCapability]:
    capability = (task.context.capability or "").strip()
    if is_orchestration_capability(
        capability,
        trigger_capabilities=orchestration_triggers,
        pipeline_capability_suffix=pipeline_capability_suffix,
    ):
        return frozenset({ExecutionCapability.ORCHESTRATION})
    return frozenset({ExecutionCapability.AGENT})


def task_result_from_agent_execution(
    task: Task,
    *,
    run_id: RunId,
    execution_result: AgentExecutionResult,
) -> TaskResult:
    state = (
        TaskState.COMPLETED
        if execution_result.status is AgentExecutionStatus.COMPLETED
        else TaskState.FAILED
    )
    return TaskResult(
        task_id=task.task_id,
        run_id=run_id,
        state=state,
        answer=execution_result.summary,
        agent_id=execution_result.agent_id,
        execution_result=execution_result,
    )


class TaskBoundAgenticDelegate:
    """Routes canonical agentic task requests through the governed agent engine."""

    __slots__ = ("_task", "_agent_engine", "_agent_router")

    def __init__(
        self,
        task: Task,
        *,
        agent_engine: AgentEnginePort,
        agent_router: AgentRouter,
    ) -> None:
        self._task = task
        self._agent_engine = agent_engine
        self._agent_router = agent_router

    async def execute(
        self,
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        del request
        run_id, _attempt_id = require_active_execution_identity()
        task = self._task
        if not task.agent_id:
            agent = self._agent_router.route(task, run_id=run_id)
            task = task.model_copy(update={"agent_id": agent.get_contract().id})
        runtime_request = task.to_runtime_request(run_id=run_id)
        execution_result = await self._agent_engine.run_with_result(runtime_request)
        return task_result_from_agent_execution(
            task,
            run_id=run_id,
            execution_result=execution_result,
        )


def build_host_task_strategy_router(
    task: Task,
    *,
    agent_engine: AgentEnginePort,
    agent_router: AgentRouter,
    orchestration_executor: OrchestrationExecutor,
) -> StrategyExecutionRouter[TaskExecutionInput, TaskResult, TaskResult]:
    return StrategyExecutionRouter[
        TaskExecutionInput,
        TaskResult,
        TaskResult,
    ](
        agent_executor=TaskBoundAgenticDelegate(
            task,
            agent_engine=agent_engine,
            agent_router=agent_router,
        ),
        orchestration_executor=TaskBoundOrchestrationDelegate(
            task,
            orchestration_executor,
        ),
    )


class HostTaskExecutionPort(Protocol):
    async def execute(
        self,
        task: Task,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult: ...


@dataclass(frozen=True, slots=True)
class HostTaskExecution:
    """Composition-root host task execution through canonical :class:`Execution`."""

    _agent_engine: AgentEnginePort
    _agent_router: AgentRouter
    _orchestration_executor: OrchestrationExecutor
    _orchestration_triggers: frozenset[str]
    _pipeline_capability_suffix: str
    _ledger_factory: ExecutionBudgetLedgerFactory | None
    _run_budget: RunBudget | None
    _terminal_publisher: HostTaskTerminalPublisher | None = None

    def _execution_runtime_for_task(
        self,
        task: Task,
    ) -> ExecutionRuntime[
        ExecutionRequest[TaskExecutionInput, TaskResult],
        TaskResult,
    ]:
        router = build_host_task_strategy_router(
            task,
            agent_engine=self._agent_engine,
            agent_router=self._agent_router,
            orchestration_executor=self._orchestration_executor,
        )
        return ExecutionRuntime[
            ExecutionRequest[TaskExecutionInput, TaskResult],
            TaskResult,
        ](
            router,
            ledger_factory=self._ledger_factory,
            run_budget=self._run_budget,
        )

    async def execute(
        self,
        task: Task,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        capabilities = resolve_task_execution_capabilities(
            task,
            orchestration_triggers=self._orchestration_triggers,
            pipeline_capability_suffix=self._pipeline_capability_suffix,
        )
        request = execution_request_from_task(
            task,
            capabilities=capabilities,
            output_type=TaskResult,
        )
        resolved_run_id = run_id or mint_run_id()
        resolved_attempt_id = attempt_id or mint_attempt_id()
        root_identity = mint_root_execution_identity(
            run_id=resolved_run_id,
            attempt_id=resolved_attempt_id,
        )
        options = RootExecutionOptions(
            authority=resolve_root_parent_execution_authority(task.execution_authority),
            tenant_id=task.tenant_id,
            run_id=root_identity.run_id,
            attempt_id=root_identity.attempt_id,
            execution_id=root_identity.execution_id,
        )
        await ActiveTaskRegistry.register(task, root_identity.run_id)
        try:
            execution = Execution(self._execution_runtime_for_task(task))
            result = await execution.execute(request, options=options)
            if (
                self._terminal_publisher is not None
                and terminal_outcome_from_task_state(result.state) is not None
            ):
                terminal_task = task.model_copy(
                    update={
                        "state": result.state,
                        "agent_id": result.agent_id or task.agent_id,
                    },
                )
                await self._terminal_publisher.publish_terminal(
                    terminal_task,
                    run_id=root_identity.run_id,
                    attempt_id=root_identity.attempt_id,
                    execution_id=root_identity.execution_id,
                )
            return result
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, root_identity.run_id)
