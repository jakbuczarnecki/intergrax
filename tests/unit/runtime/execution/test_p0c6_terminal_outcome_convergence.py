# © Artur Czarnecki. All rights reserved.

"""P0C-6 — durable terminal outcome convergence proofs."""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
)
from intergrax.runtime.cancellation.resume_admission import (
    CheckpointNotResumableError,
    assert_checkpoint_resumable,
    is_checkpoint_resumable,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
    normalize_terminal_record,
    reconcile_task_state_with_terminal_outcome,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduler import LongRunningScheduler, UnifiedTaskResumeExecutor
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-p0c6"


def _paused_checkpoint(*, task_id: str | None = None) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-p0c6"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        resume_token="rt-p0c6",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


def test_completed_terminal_is_durable() -> None:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    record = terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    assert record.outcome is ExecutionTerminalOutcome.COMPLETED
    loaded = terminal.get_terminal_record(tenant_id=_TENANT, task_id=task_id)
    assert loaded is not None
    assert loaded.outcome is ExecutionTerminalOutcome.COMPLETED
    assert loaded.run_id == run_id


def test_failed_terminal_is_durable() -> None:
    task_id = str(mint_task_id())
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    record = terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.FAILED,
        reason="graph_failed",
    )
    assert record.outcome is ExecutionTerminalOutcome.FAILED
    assert terminal.get_terminal_record(tenant_id=_TENANT, task_id=task_id) == record


def test_cancelled_terminal_regression() -> None:
    task_id = str(mint_task_id())
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    record = terminal.record_cancellation(tenant_id=_TENANT, task_id=task_id, reason="operator_cancel")
    assert record.outcome is ExecutionTerminalOutcome.CANCELLED


def test_same_terminal_outcome_is_idempotent() -> None:
    task_id = str(mint_task_id())
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    first = terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    second = terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="late_duplicate",
    )
    assert second.outcome == first.outcome
    assert second.recorded_at_utc == first.recorded_at_utc


def test_conflicting_terminal_transition_is_rejected() -> None:
    task_id = str(mint_task_id())
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    with pytest.raises(ExecutionTerminalConflictError) as exc_info:
        terminal.commit_terminal_outcome(
            tenant_id=_TENANT,
            task_id=task_id,
            outcome=ExecutionTerminalOutcome.FAILED,
            reason="graph_failed",
        )
    assert exc_info.value.existing_outcome is ExecutionTerminalOutcome.COMPLETED


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (ExecutionTerminalOutcome.COMPLETED, ExecutionTerminalOutcome.FAILED),
        (ExecutionTerminalOutcome.COMPLETED, ExecutionTerminalOutcome.CANCELLED),
        (ExecutionTerminalOutcome.FAILED, ExecutionTerminalOutcome.CANCELLED),
    ],
    ids=["completed_vs_failed", "completed_vs_cancelled", "failed_vs_cancelled"],
)
def test_concurrent_different_terminal_outcomes_have_one_winner(
    first: ExecutionTerminalOutcome,
    second: ExecutionTerminalOutcome,
) -> None:
    store = InMemoryExecutionTerminalStore()
    terminal = ExecutionTerminalService(store)
    task_id = str(mint_task_id())
    barrier = threading.Barrier(2)
    winners: list[ExecutionTerminalRecord] = []
    conflicts: list[ExecutionTerminalConflictError] = []

    def worker(outcome: ExecutionTerminalOutcome) -> None:
        barrier.wait()
        try:
            winners.append(
                terminal.commit_terminal_outcome(
                    tenant_id=_TENANT,
                    task_id=task_id,
                    outcome=outcome,
                    reason=outcome.value,
                ),
            )
        except ExecutionTerminalConflictError as exc:
            conflicts.append(exc)

    t1 = threading.Thread(target=worker, args=(first,))
    t2 = threading.Thread(target=worker, args=(second,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(winners) == 1
    assert len(conflicts) == 1
    canonical = terminal.get_terminal_record(tenant_id=_TENANT, task_id=task_id)
    assert canonical is not None
    assert canonical.outcome in {first, second}
    assert conflicts[0].existing_outcome == canonical.outcome


def test_terminal_state_survives_process_restart(tmp_path) -> None:
    db_path = tmp_path / "terminal.db"
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    terminal_a = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    terminal_a.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.FAILED,
        reason="graph_failed",
    )

    restarted = SQLiteTaskCheckpointStore(db_path=db_path)
    terminal_b = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(restarted))
    loaded = terminal_b.get_terminal_record(tenant_id=_TENANT, task_id=task_id)
    assert loaded is not None
    assert loaded.outcome is ExecutionTerminalOutcome.FAILED
    with pytest.raises(ExecutionTerminalConflictError):
        terminal_b.commit_terminal_outcome(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            outcome=ExecutionTerminalOutcome.COMPLETED,
            reason="completed",
        )


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (ExecutionTerminalOutcome.COMPLETED, "completed"),
        (ExecutionTerminalOutcome.FAILED, "graph_failed"),
    ],
    ids=["completed", "failed"],
)
def test_terminal_outcome_blocks_stale_checkpoint_resume(
    outcome: ExecutionTerminalOutcome,
    reason: str,
) -> None:
    checkpoint = _paused_checkpoint()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        reason=reason,
        outcome=outcome,
    )
    assert is_checkpoint_resumable(checkpoint, execution_terminal=terminal) is False
    with pytest.raises(CheckpointNotResumableError, match=outcome.value):
        assert_checkpoint_resumable(checkpoint, execution_terminal=terminal)


def test_retryable_failure_does_not_commit_failed() -> None:
    coordinator = RetryCoordinator(
        max_run_retries=2,
        retry_run_on=frozenset({RuntimeErrorCode.VALIDATION_ERROR}),
    )
    assert coordinator.should_retry_run(attempt=0, error_code=RuntimeErrorCode.VALIDATION_ERROR) is True
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    assert terminal.get_terminal_record(tenant_id=_TENANT, task_id=str(mint_task_id())) is None


def test_terminal_store_failure_prevents_fake_terminal_success() -> None:
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal._store.put_if_absent = MagicMock(return_value=False)  # type: ignore[method-assign]
    terminal._store.load_record = MagicMock(return_value=None)  # type: ignore[method-assign]
    with pytest.raises(ExecutionTerminalError, match="race lost"):
        terminal.commit_terminal_outcome(
            tenant_id=_TENANT,
            task_id=str(mint_task_id()),
            outcome=ExecutionTerminalOutcome.COMPLETED,
            reason="completed",
            production_mode=False,
        )


def test_tenant_isolation() -> None:
    task_id = str(mint_task_id())
    checkpoint_a = _paused_checkpoint(task_id=task_id)
    checkpoint_b = TaskCheckpoint(
        task_id=task_id,
        tenant_id="tenant-other",
        resume_token="rt-other",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=checkpoint_a.task_snapshot,
    )
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    assert is_checkpoint_resumable(checkpoint_a, execution_terminal=terminal) is False
    assert is_checkpoint_resumable(checkpoint_b, execution_terminal=terminal) is True


def test_run_identity_conflict() -> None:
    task_id = str(mint_task_id())
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_r1,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    with pytest.raises(ExecutionTerminalConflictError, match="run_id mismatch"):
        terminal.commit_terminal_outcome(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_r2,
            outcome=ExecutionTerminalOutcome.COMPLETED,
            reason="completed",
        )


def test_corrupt_terminal_record_fails_closed() -> None:
    checkpoint = _paused_checkpoint()

    class _CorruptTerminalStore(InMemoryExecutionTerminalStore):
        def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
            return ExecutionTerminalRecord(
                tenant_id=tenant_id,
                task_id=task_id,
                outcome=ExecutionTerminalOutcome.CANCELLED,
                reason="",
                recorded_at_utc="",
            )

    terminal = ExecutionTerminalService(_CorruptTerminalStore())
    assert is_checkpoint_resumable(checkpoint, execution_terminal=terminal) is False


def test_legacy_cancelled_payload_remains_readable() -> None:
    record = ExecutionTerminalRecord(
        tenant_id=_TENANT,
        task_id=str(mint_task_id()),
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
        recorded_at_utc="2026-01-01T00:00:00+00:00",
    )
    normalized = normalize_terminal_record(record)
    assert normalized.outcome is ExecutionTerminalOutcome.CANCELLED


@pytest.mark.asyncio
async def test_scheduler_skips_resume_after_completed_terminal(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    terminal = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    runner = AsyncMock()
    scheduler = LongRunningScheduler(
        store,
        UnifiedTaskResumeExecutor(runner),
        schedule_store=store,
        ledger=store,
        execution_terminal=terminal,
    )
    from intergrax.runtime.long_running.scheduled_resume import ScheduledResume

    entry = ScheduledResume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc="2000-01-01T00:00:00+00:00",
    )
    store.schedule(entry)
    processed = await scheduler.tick(
        now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc),
    )
    assert processed == 0
    runner.run_task.assert_not_called()


def test_reconcile_task_state_preserves_partial_when_canonical_completed() -> None:
    reconciled = reconcile_task_state_with_terminal_outcome(
        TaskState.PARTIALLY_COMPLETED,
        ExecutionTerminalOutcome.COMPLETED,
    )
    assert reconciled is TaskState.PARTIALLY_COMPLETED


def test_reconcile_task_state_projects_conflicting_outcome() -> None:
    assert (
        reconcile_task_state_with_terminal_outcome(
            TaskState.COMPLETED,
            ExecutionTerminalOutcome.CANCELLED,
        )
        is TaskState.CANCELLED
    )
    assert (
        reconcile_task_state_with_terminal_outcome(
            TaskState.COMPLETED,
            ExecutionTerminalOutcome.FAILED,
        )
        is TaskState.FAILED
    )
    assert (
        reconcile_task_state_with_terminal_outcome(
            TaskState.FAILED,
            ExecutionTerminalOutcome.COMPLETED,
        )
        is TaskState.COMPLETED
    )


def test_reconcile_task_state_keeps_same_exact_state() -> None:
    assert (
        reconcile_task_state_with_terminal_outcome(
            TaskState.FAILED,
            ExecutionTerminalOutcome.FAILED,
        )
        is TaskState.FAILED
    )


def _nexus_finish_loop(terminal: ExecutionTerminalService) -> NexusLoop:
    return NexusLoop(AgentRegistry(), execution_terminal=terminal)


async def _finish_terminal_task(
    loop: NexusLoop,
    task: Task,
    *,
    run_id,
    attempt_id,
):
    trace_emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id)
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        return await loop._finish_task(  # noqa: SLF001
            task,
            trace_emitter,
            answer="done",
            executions=[],
            validation=ValidationResult(valid=True),
            plan=None,
            retry_records=[],
            graph_id="graph-p0c6",
        )
    finally:
        reset_active_execution_identity(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("canonical_outcome", "expected_state", "forbidden_event"),
    [
        (ExecutionTerminalOutcome.CANCELLED, TaskState.CANCELLED, RuntimeEventType.TASK_COMPLETED),
        (ExecutionTerminalOutcome.FAILED, TaskState.FAILED, RuntimeEventType.TASK_COMPLETED),
    ],
    ids=["cancelled", "failed"],
)
async def test_finish_reconciles_completed_to_canonical_outcome(
    canonical_outcome: ExecutionTerminalOutcome,
    expected_state: TaskState,
    forbidden_event: RuntimeEventType,
) -> None:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=canonical_outcome,
        reason=canonical_outcome.value,
    )
    loop = _nexus_finish_loop(terminal)
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.COMPLETED,
    )
    result = await _finish_terminal_task(loop, task, run_id=run_id, attempt_id=attempt_id)
    assert task.state is expected_state
    assert result.state is expected_state
    assert not any(event.event_type is forbidden_event for event in loop.event_bus.history)


@pytest.mark.asyncio
async def test_finish_reconciles_failed_to_canonical_completed() -> None:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    loop = _nexus_finish_loop(terminal)
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.FAILED,
    )
    result = await _finish_terminal_task(loop, task, run_id=run_id, attempt_id=attempt_id)
    assert task.state is TaskState.COMPLETED
    assert result.state is TaskState.COMPLETED
    assert not any(
        event.event_type is RuntimeEventType.TASK_FAILED for event in loop.event_bus.history
    )


@pytest.mark.asyncio
async def test_finish_preserves_partial_when_canonical_completed() -> None:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
    )
    loop = _nexus_finish_loop(terminal)
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.PARTIALLY_COMPLETED,
    )
    result = await _finish_terminal_task(loop, task, run_id=run_id, attempt_id=attempt_id)
    assert task.state is TaskState.PARTIALLY_COMPLETED
    assert result.state is TaskState.PARTIALLY_COMPLETED


@pytest.mark.asyncio
async def test_finish_run_mismatch_fails_closed() -> None:
    task_id = str(mint_task_id())
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    attempt_id = mint_attempt_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_r1,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )
    loop = _nexus_finish_loop(terminal)
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.COMPLETED,
    )
    with pytest.raises(ExecutionTerminalConflictError, match="run_id mismatch"):
        await _finish_terminal_task(loop, task, run_id=run_r2, attempt_id=attempt_id)


@pytest.mark.asyncio
async def test_losing_terminal_event_not_published() -> None:
    task_id = str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.commit_terminal_outcome(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.CANCELLED,
        reason="operator_cancel",
    )
    loop = _nexus_finish_loop(terminal)
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.COMPLETED,
    )
    await _finish_terminal_task(loop, task, run_id=run_id, attempt_id=attempt_id)
    terminal_events = [
        event.event_type
        for event in loop.event_bus.history
        if event.event_type
        in {
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.TASK_CANCELLED,
            RuntimeEventType.TASK_FAILED,
        }
    ]
    assert terminal_events == []


def test_terminal_reconciliation_store_failure_fails_closed() -> None:
    task = Task(
        task_id=str(mint_task_id()),
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.COMPLETED,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )

    class _ConflictWithoutCanonicalStore(InMemoryExecutionTerminalStore):
        def __init__(self) -> None:
            super().__init__()
            self._seeded = False

        def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
            if not self._seeded:
                self._seeded = True
                return ExecutionTerminalRecord(
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                    outcome=ExecutionTerminalOutcome.CANCELLED,
                    reason="operator_cancel",
                    recorded_at_utc="2026-01-01T00:00:00+00:00",
                )
            return None

    terminal = ExecutionTerminalService(_ConflictWithoutCanonicalStore())
    loop = MagicMock(spec=NexusLoop)
    loop._production_mode = False
    loop._execution_terminal = terminal
    loop._commit_durable_terminal_authority = NexusLoop._commit_durable_terminal_authority.__get__(loop)
    try:
        with pytest.raises(ExecutionTerminalError, match="without canonical record"):
            loop._commit_durable_terminal_authority(task)
    finally:
        reset_active_execution_identity(token)


def test_nexus_finish_task_conflict_reconciles_canonical_authority() -> None:
    from intergrax.runtime.nexus.nexus_loop import NexusLoop

    task = Task(
        task_id=str(mint_task_id()),
        tenant_id=_TENANT,
        user_id="user",
        message="done",
        state=TaskState.COMPLETED,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    loop = MagicMock(spec=NexusLoop)
    loop._production_mode = False
    loop._execution_terminal = terminal
    loop._commit_durable_terminal_authority = NexusLoop._commit_durable_terminal_authority.__get__(loop)
    try:
        terminal.commit_terminal_outcome(
            tenant_id=_TENANT,
            task_id=task.task_id,
            run_id=run_id,
            outcome=ExecutionTerminalOutcome.CANCELLED,
            reason="operator_cancel",
        )
        task.state = TaskState.COMPLETED
        resolution = loop._commit_durable_terminal_authority(task)
        assert resolution.should_publish_terminal_event is False
        assert resolution.canonical_record is not None
        assert resolution.canonical_record.outcome is ExecutionTerminalOutcome.CANCELLED
        assert task.state is TaskState.CANCELLED
    finally:
        reset_active_execution_identity(token)
