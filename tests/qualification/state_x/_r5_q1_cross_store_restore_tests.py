# © Artur Czarnecki. All rights reserved.

"""STATE-X-R5-Q1 — cross-store restore skew behavioral closure (FRZ-REC-08)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from intergrax.agents.persistence.compensation_queue_worker import drain_pending_compensation_jobs
from intergrax.agents.persistence.compensation_queue_store import (
    CompensationJobStatus,
    SQLiteCompensationQueueStore,
)
from intergrax.agents.persistence.compensation_side_effect_input import (
    compensation_side_effect_input_from_job,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
)
from intergrax.contracts.idempotency_store import InvocationStatus
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeValidationError,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import TaskState
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from tests.qualification.state_x._r3_r2_support import (
    _TENANT_A,
    _build_canonical_compensation_production_execution,
    _execute_compensation_with_lab_governance,
    _sample_compensation_job,
)
from tests.qualification.state_x._r5_backup_restore_support import (
    R5_BACKUP_RESTORE_FAMILY_MATRIX,
    run_trace_excluded_from_execution_recovery_authority_sources,
    run_trace_not_resume_consumer,
)
from tests.qualification.state_x.test_state_x_r1_checkpoint_resume_terminal import (
    _paused_checkpoint,
    _resume_task,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sqlite_checkpoint_store(path: Path) -> SQLiteTaskCheckpointStore:
    return SQLiteTaskCheckpointStore(db_path=path)


def _sqlite_snapshot_file(source_db: Path, snapshot_path: Path) -> None:
    if snapshot_path.exists():
        snapshot_path.unlink()
    with sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True) as src:
        with sqlite3.connect(snapshot_path) as dst:
            src.backup(dst)


def _sqlite_restore_into(snapshot_path: Path, live_db: Path) -> None:
    with sqlite3.connect(f"file:{snapshot_path.as_posix()}?mode=ro", uri=True) as src:
        with sqlite3.connect(live_db) as dst:
            src.backup(dst)


def test_r5_q1_old_checkpoint_newer_terminal_restore_skew_fails_closed(
    tmp_path: Path,
) -> None:
    """T1 resumable checkpoint (SQLite) + T2 terminal COMPLETED (separate store) → no resurrection."""
    checkpoint = _paused_checkpoint()
    assert checkpoint.runtime is not None
    cp_db = tmp_path / "checkpoint-t1.db"
    checkpoint_store = _sqlite_checkpoint_store(cp_db)
    checkpoint_store.save(checkpoint)

    terminal_store = InMemoryExecutionTerminalStore()
    terminal = ExecutionTerminalService(terminal_store)
    terminal.commit_terminal_outcome(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        run_id=checkpoint.runtime.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="done-at-t2",
    )

    task = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    before = task.model_dump()

    with pytest.raises(CheckpointResumeValidationError):
        LongRunningCoordinator.restore_if_resuming(
            task,
            checkpoint_store,
            execution_terminal=terminal,
        )

    assert task.model_dump() == before
    assert task.state is TaskState.CREATED


def test_r5_q1_terminal_backend_corruption_is_not_treated_as_absence(
    tmp_path: Path,
) -> None:
    checkpoint = _paused_checkpoint()
    store = _sqlite_checkpoint_store(tmp_path / "cp-corrupt-term.db")
    store.save(checkpoint)
    task = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    before = task.model_dump()

    class _CorruptTerminalStore(InMemoryExecutionTerminalStore):
        def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
            raise ExecutionTerminalError("corrupt terminal bytes")

    terminal = ExecutionTerminalService(_CorruptTerminalStore())

    with pytest.raises(CheckpointResumeValidationError):
        LongRunningCoordinator.restore_if_resuming(
            task,
            store,
            execution_terminal=terminal,
        )
    assert task.model_dump() == before


@pytest.mark.asyncio
async def test_r5_q1_old_compensation_queue_newer_idempotency_no_duplicate_effect(
    tmp_path: Path,
) -> None:
    """T1 queue backup (PENDING) + T2 effect/idempotency COMPLETED + T3 queue restore → T4 drain: no 2nd effect."""
    comp_path = tmp_path / "compensation.db"
    idem_path = tmp_path / "idempotency.db"
    queue = SQLiteCompensationQueueStore(comp_path)
    idem_store = SQLiteIdempotencyStore(str(idem_path))
    execution, handler, _catalog = _build_canonical_compensation_production_execution(idem_store)

    job = _sample_compensation_job(key_suffix="r5-q1-skew")
    queue.enqueue(job)
    queue_t1_backup = tmp_path / "compensation-t1.bak"
    _sqlite_snapshot_file(comp_path, queue_t1_backup)

    claim = queue.claim_pending(_TENANT_A, "worker-t2", lease_seconds=30, limit=1)[0]
    work = compensation_side_effect_input_from_job(claim.job)
    assert (await _execute_compensation_with_lab_governance(execution, work)).status == "success"
    assert handler.calls == 1
    assert idem_store.get_status(_TENANT_A, work.idempotency_key) == InvocationStatus.COMPLETED
    queue.complete_claim(claim)
    loaded_after_t2 = queue.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert loaded_after_t2 is not None
    assert loaded_after_t2.status == CompensationJobStatus.COMPLETED

    _sqlite_restore_into(queue_t1_backup, comp_path)
    restored_queue = SQLiteCompensationQueueStore(comp_path)
    restored_job = restored_queue.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert restored_job is not None
    assert restored_job.status == CompensationJobStatus.PENDING
    assert idem_store.get_status(_TENANT_A, work.idempotency_key) == InvocationStatus.COMPLETED

    drained = await drain_pending_compensation_jobs(
        restored_queue,
        tenant_id=_TENANT_A,
        side_effect_execution=execution,
        limit=10,
        owner_id="r5-q1-drain",
        lease_seconds=30,
    )
    assert handler.calls == 1
    assert len(drained) >= 1
    final_queue = restored_queue.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert final_queue is not None
    assert final_queue.status != CompensationJobStatus.PENDING


def test_r5_q1_run_trace_loss_is_non_authoritative_not_claimed_rebuildable() -> None:
    assert run_trace_not_resume_consumer()
    assert run_trace_excluded_from_execution_recovery_authority_sources()
    row = next(r for r in R5_BACKUP_RESTORE_FAMILY_MATRIX if r.family_id == "SX-F08")
    blob = " ".join((row.physical_backup_unit, row.supported_restore_semantics)).lower()
    assert "rebuild" not in blob or "not certified" in blob or "not guaranteed" in blob
    assert "non-authoritative" in blob or "observability" in blob
