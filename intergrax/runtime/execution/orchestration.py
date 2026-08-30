# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Orchestration execution backend routed through Nexus (UE-9D)."""

from __future__ import annotations

from typing import Protocol, TypeVar

from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    mint_attempt_id,
    mint_run_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.budget.ledger import ExecutionBudgetLedgerFactory
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
    RootTaskIdentity,
    mint_root_execution_identity,
)
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.execution.task_adapter import TaskExecutionInput, execution_request_from_task
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_task,
    prepare_task_for_checkpoint_resume,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import execution_identity_from_checkpoint
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult

_ORCHESTRATION_CAPABILITIES = frozenset({ExecutionCapability.ORCHESTRATION})

OutputT = TypeVar("OutputT")


class NexusOrchestrationPort(Protocol):
    async def handle_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        ...


def resolve_root_task_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    resume_checkpoint: TaskCheckpoint | None = None,
) -> RootTaskIdentity:
    if resume_checkpoint is not None:
        checkpoint_run_id, checkpoint_attempt_id = execution_identity_from_checkpoint(
            resume_checkpoint
        )
        if run_id is not None and run_id != checkpoint_run_id:
            raise ValueError(
                "explicit run_id conflicts with resume checkpoint identity: "
                f"{run_id!r} != {checkpoint_run_id!r}"
            )
        resolved_run_id = checkpoint_run_id
        resolved_attempt_id = attempt_id or checkpoint_attempt_id
    else:
        resolved_run_id = run_id or mint_run_id()
        resolved_attempt_id = attempt_id or mint_attempt_id()
    return mint_root_execution_identity(
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
    )


class OrchestrationExecutor:
    """Orchestration execution backend behind :class:`ExecutionBoundary`."""

    __slots__ = ("_nexus_loop",)

    def __init__(self, nexus_loop: NexusOrchestrationPort) -> None:
        self._nexus_loop = nexus_loop

    async def execute(self, task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        require_active_execution_id()

        return await self._nexus_loop.handle_task(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
        )


class TaskBoundOrchestrationDelegate:
    """Routes canonical orchestration requests to Nexus using the bound root Task."""

    __slots__ = ("_task", "_executor")

    def __init__(self, task: Task, executor: OrchestrationExecutor) -> None:
        self._task = task
        self._executor = executor

    async def execute(
        self,
        request: ExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        del request
        return await self._executor.execute(self._task)


async def execute_root_task(
    task: Task,
    *,
    nexus_loop: NexusLoop,
    identity: RootTaskIdentity,
    resume_checkpoint: TaskCheckpoint | None = None,
    ledger_factory: ExecutionBudgetLedgerFactory | None = None,
    run_budget: RunBudget | None = None,
) -> TaskResult:
    if resume_checkpoint is not None and resume_checkpoint.runtime is not None:
        _checkpoint_run_id, checkpoint_attempt_id = execution_identity_from_checkpoint(
            resume_checkpoint
        )
        if identity.attempt_id != checkpoint_attempt_id:
            prepare_task_for_checkpoint_resume(
                task,
                resume_checkpoint,
                active_attempt_id=identity.attempt_id,
                active_root_execution_id=identity.execution_id,
            )
        elif task.runtime.orchestration.runtime_checkpoint is None:
            apply_runtime_checkpoint_to_task(task, resume_checkpoint.runtime)

    request = execution_request_from_task(
        task,
        capabilities=_ORCHESTRATION_CAPABILITIES,
        output_type=TaskResult,
    )
    router = StrategyExecutionRouter[
        TaskExecutionInput,
        TaskResult,
        TaskResult,
    ](
        orchestration_executor=TaskBoundOrchestrationDelegate(
            task,
            OrchestrationExecutor(nexus_loop),
        ),
    )
    runtime = ExecutionRuntime[
        ExecutionRequest[TaskExecutionInput, TaskResult],
        TaskResult,
    ](
        router,
        ledger_factory=ledger_factory,
        run_budget=run_budget,
    )
    root_context = RootExecutionContext(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        authority=resolve_root_parent_execution_authority(task.execution_authority),
        tenant_id=task.tenant_id,
    )
    return await runtime.execute(request, root_context)
