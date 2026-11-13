# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for the PDC-X B-D7 qualification package (real SQLite, real scheduler)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskLongRunningOptions,
)
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.utils.time_provider import SystemTimeProvider

TENANT_ID = "t-d7"
RESUME_TOKEN = "token-d7"


def paused_checkpoint(*, tenant_id: str = TENANT_ID) -> TaskCheckpoint:
    """Paused long-running checkpoint that stays resumable across repeated resumes."""
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        tenant_id=tenant_id,
        user_id="u1",
        message="recurring",
        task_id=task_id,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.state = TaskState.WAITING_FOR_RESOURCES
    runtime = RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=mint_execution_id(),
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                ),
            ],
        ),
    )
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token=RESUME_TOKEN,
        task_state=TaskState.WAITING_FOR_RESOURCES,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="awaiting",
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
        runtime=runtime,
    )


def ok_task_result(task_id: str = "task-1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        state=TaskState.COMPLETED,
        success=True,
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


class CountingResumeExecutor:
    """Real ``TaskResumeExecutor`` port implementation that records every resume."""

    def __init__(self) -> None:
        self.schedule_ids: list[str] = []
        self.occurrences: list[str] = []

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        self.schedule_ids.append(str(task.metadata.get("schedule_id")))
        self.occurrences.append(str(task.metadata.get("recurrence_occurrence_utc")))
        return ok_task_result(checkpoint.task_id)


def build_scheduler(
    db_path: Path,
    executor: CountingResumeExecutor,
    *,
    owner_id: str = "sched-d7",
    with_recurrence: bool = True,
) -> tuple[LongRunningScheduler, SQLiteTaskCheckpointStore, RecurrenceMaterializer]:
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    materializer = RecurrenceMaterializer(store)
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,
        owner_id=owner_id,
        recurrence=materializer if with_recurrence else None,
    )
    return scheduler, store, materializer


async def drain_ticks(
    scheduler: LongRunningScheduler,
    *,
    now: datetime,
    max_ticks: int = 25,
) -> int:
    """Tick at a frozen ``now`` until idle (bounded); returns total resumes."""
    total = 0
    for _ in range(max_ticks):
        fired = await scheduler.tick(now=now)
        if fired == 0:
            return total
        total += fired
    raise AssertionError("scheduler did not become idle within the tick bound")
