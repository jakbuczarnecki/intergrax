# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R2-H2 — durable checkpoint revision & stale writer protection."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeEligibility,
    CheckpointResumeValidationError,
    validate_checkpoint_not_stale,
)
from intergrax.runtime.long_running.checkpoint_revision import (
    CheckpointIdConflictError,
    CheckpointRevisionRequiredError,
    StaleCheckpointWriteError,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-h2"


def _paused_checkpoint(
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    checkpoint_id: str | None = None,
    resume_token: str = "rt-h2",
    created_at_utc: str = "2026-09-10T12:00:00+00:00",
    progress_message: str = "paused",
    execution_authority: ParentExecutionAuthority | None = None,
) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    task = Task(
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=execution_authority,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token=resume_token),
        ),
    )
    return TaskCheckpoint(
        checkpoint_id=checkpoint_id or f"ckpt_{mint_task_id()}",
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        progress_message=progress_message,
        created_at_utc=created_at_utc,
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def _save_next(
    store: SQLiteTaskCheckpointStore,
    checkpoint: TaskCheckpoint,
    *,
    base: TaskCheckpoint,
) -> TaskCheckpoint:
    return store.save(checkpoint, expected_revision=base.revision)


def test_first_write_assigns_revision_one(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "first.db")
    saved = store.save(_paused_checkpoint())
    assert saved.revision == 1
    assert saved.store_sequence is not None


def test_next_write_increments_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "next.db")
    first = store.save(_paused_checkpoint())
    second = _save_next(
        store,
        first.model_copy(update={"checkpoint_id": "ckpt_2", "progress_message": "step-2"}),
        base=first,
    )
    assert first.revision == 1
    assert second.revision == 2


def test_stale_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale.db")
    first = store.save(_paused_checkpoint())
    _save_next(
        store,
        first.model_copy(update={"checkpoint_id": "ckpt_2", "progress_message": "v2"}),
        base=first,
    )
    with pytest.raises(StaleCheckpointWriteError) as exc_info:
        store.save(
            first.model_copy(update={"checkpoint_id": "ckpt_stale", "progress_message": "stale-v1"}),
            expected_revision=first.revision,
        )
    assert exc_info.value.expected_revision == 1
    assert exc_info.value.actual_revision == 2


def test_concurrent_same_predecessor_exactly_one_success(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "race.db")
    base = store.save(_paused_checkpoint())
    barrier = threading.Barrier(2)
    results: list[tuple[str, int | None]] = []

    def writer(name: str) -> None:
        barrier.wait()
        checkpoint = base.model_copy(
            update={"checkpoint_id": f"ckpt_{name}", "progress_message": name},
        )
        try:
            saved = store.save(checkpoint, expected_revision=base.revision)
            results.append(("ok", saved.revision))
        except StaleCheckpointWriteError:
            results.append(("stale", None))

    threads = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [("ok", 2), ("stale", None)]
    latest = store.get_latest(base.task_id, _TENANT)
    assert latest is not None
    assert latest.revision == 2


def test_revision_cannot_fork(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "fork.db")
    base = store.save(_paused_checkpoint())
    first_success = store.save(
        base.model_copy(update={"checkpoint_id": "ckpt_a", "progress_message": "a"}),
        expected_revision=base.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(update={"checkpoint_id": "ckpt_b", "progress_message": "b"}),
            expected_revision=base.revision,
        )
    assert first_success.revision == 2


def test_revision_cannot_skip(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "skip.db")
    base = store.save(_paused_checkpoint())
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(update={"checkpoint_id": "ckpt_skip", "progress_message": "skip"}),
            expected_revision=(base.revision or 0) + 1,
        )


def test_late_stale_writer_blocker_reproduction(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "late.db")
    v1 = store.save(_paused_checkpoint(progress_message="v1"))
    snapshot_v1 = v1
    v2 = _save_next(
        store,
        v1.model_copy(update={"checkpoint_id": "ckpt_v2", "progress_message": "v2"}),
        base=v1,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            snapshot_v1.model_copy(
                update={"checkpoint_id": "ckpt_stale_v1", "progress_message": "stale-v1"},
            ),
            expected_revision=snapshot_v1.revision,
        )
    latest = store.get_latest(v1.task_id, _TENANT)
    assert latest is not None
    assert latest.progress_message == v2.progress_message


def test_same_timestamp_stale_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "same-ts.db")
    ts = "2026-09-10T12:00:00+00:00"
    first = store.save(_paused_checkpoint(created_at_utc=ts))
    _save_next(
        store,
        first.model_copy(
            update={"checkpoint_id": "ckpt_2", "created_at_utc": ts, "progress_message": "v2"},
        ),
        base=first,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            first.model_copy(
                update={
                    "checkpoint_id": "ckpt_stale",
                    "created_at_utc": ts,
                    "progress_message": "stale",
                },
            ),
            expected_revision=first.revision,
        )


def test_missing_timestamp_stale_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "missing-ts.db")
    first = store.save(_paused_checkpoint(created_at_utc=""))
    _save_next(
        store,
        first.model_copy(
            update={"checkpoint_id": "ckpt_2", "created_at_utc": "", "progress_message": "v2"},
        ),
        base=first,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            first.model_copy(
                update={"checkpoint_id": "ckpt_stale", "created_at_utc": "", "progress_message": "stale"},
            ),
            expected_revision=first.revision,
        )


def test_manipulated_timestamp_stale_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "manip-ts.db")
    first = store.save(_paused_checkpoint(created_at_utc="2026-09-10T10:00:00+00:00"))
    _save_next(
        store,
        first.model_copy(
            update={
                "checkpoint_id": "ckpt_2",
                "created_at_utc": "2026-09-10T08:00:00+00:00",
                "progress_message": "v2",
            },
        ),
        base=first,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            first.model_copy(
                update={
                    "checkpoint_id": "ckpt_stale",
                    "created_at_utc": "2026-09-10T20:00:00+00:00",
                    "progress_message": "stale",
                },
            ),
            expected_revision=first.revision,
        )


def test_higher_rowid_lower_revision_not_latest(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "rowid.db")
    checkpoint = _paused_checkpoint(checkpoint_id="ckpt_seed")
    payload = (
        checkpoint.task_id,
        checkpoint.tenant_id,
        checkpoint.resume_token,
        checkpoint.task_state.value,
        "{}",
        "canonical",
        None,
        "2026-09-10T12:00:00+00:00",
        None,
    )
    with sqlite3.connect(tmp_path / "rowid.db") as conn:
        conn.execute(
            """
            INSERT INTO task_checkpoints (
                checkpoint_id, task_id, tenant_id, resume_token, task_state,
                task_snapshot_json, progress_message, notify_channel, created_at_utc,
                runtime_checkpoint_json, checkpoint_revision
            ) VALUES ('ckpt_rev2', ?, ?, ?, ?, ?, ?, ?, ?, ?, 2)
            """,
            payload,
        )
        conn.execute(
            """
            INSERT INTO task_checkpoints (
                checkpoint_id, task_id, tenant_id, resume_token, task_state,
                task_snapshot_json, progress_message, notify_channel, created_at_utc,
                runtime_checkpoint_json, checkpoint_revision
            ) VALUES ('ckpt_rev1', ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            payload,
        )
    latest = store.get_latest(checkpoint.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == "ckpt_rev2"
    assert latest.revision == 2


def test_get_latest_returns_highest_logical_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "latest.db")
    first = store.save(_paused_checkpoint())
    second = _save_next(
        store,
        first.model_copy(update={"checkpoint_id": "ckpt_2"}),
        base=first,
    )
    latest = store.get_latest(first.task_id, _TENANT)
    assert latest == second


def test_get_by_token_returns_highest_logical_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "token.db")
    token = "shared-token"
    first = store.save(_paused_checkpoint(resume_token=token))
    second = _save_next(
        store,
        first.model_copy(update={"checkpoint_id": "ckpt_2", "progress_message": "newer"}),
        base=first,
    )
    loaded = store.get_by_token(first.task_id, _TENANT, token)
    assert loaded is not None
    assert loaded.checkpoint_id == second.checkpoint_id
    assert loaded.revision == 2


def test_old_token_resume_blocked_by_validator(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "old-token.db")
    older = store.save(_paused_checkpoint(checkpoint_id="ckpt_old", resume_token="rt-old"))
    newer = _save_next(
        store,
        _paused_checkpoint(
            task_id=older.task_id,
            checkpoint_id="ckpt_new",
            resume_token="rt-new",
        ),
        base=older,
    )
    task = Task(
        task_id=older.task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="resume",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token=older.resume_token),
        ),
    )
    with pytest.raises(CheckpointResumeValidationError) as exc_info:
        LongRunningCoordinator.restore_if_resuming(task, store)
    assert exc_info.value.result.eligibility is CheckpointResumeEligibility.REJECT_STALE
    latest = store.get_latest(older.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == newer.checkpoint_id


def test_save_return_includes_revision_and_store_sequence(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "return.db")
    saved = store.save(_paused_checkpoint())
    assert saved.revision == 1
    assert saved.store_sequence is not None


def test_duplicate_same_checkpoint_id_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "dup.db")
    checkpoint = _paused_checkpoint(checkpoint_id="ckpt_same")
    first = store.save(checkpoint)
    second = store.save(checkpoint, expected_revision=first.revision)
    assert second.revision == first.revision
    assert second.store_sequence == first.store_sequence
    assert store.list_for_task(checkpoint.task_id, _TENANT) == [first]


def test_same_checkpoint_id_different_payload_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "conflict.db")
    checkpoint = store.save(_paused_checkpoint(checkpoint_id="ckpt_same"))
    conflict = checkpoint.model_copy(update={"progress_message": "different"})
    with pytest.raises(CheckpointIdConflictError):
        store.save(conflict, expected_revision=checkpoint.revision)


def test_unknown_commit_retry_does_not_create_extra_revision(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "retry.db")
    base = store.save(_paused_checkpoint())
    successor = base.model_copy(update={"checkpoint_id": "ckpt_2", "progress_message": "v2"})
    committed = store.save(successor, expected_revision=base.revision)
    retried = store.save(successor, expected_revision=base.revision)
    assert retried.revision == committed.revision == 2
    assert len(store.list_for_task(base.task_id, _TENANT)) == 2


def test_tenant_isolation(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "tenant.db")
    task_id = str(mint_task_id())
    tenant_a = store.save(_paused_checkpoint(task_id=task_id, tenant_id="tenant-a"))
    tenant_b = store.save(_paused_checkpoint(task_id=task_id, tenant_id="tenant-b"))
    assert tenant_a.revision == 1
    assert tenant_b.revision == 1
    assert store.get_latest(task_id, "tenant-a") == tenant_a
    assert store.get_latest(task_id, "tenant-b") == tenant_b


def test_two_tasks_independent_streams(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "tasks.db")
    first = store.save(_paused_checkpoint())
    second = store.save(_paused_checkpoint())
    assert first.revision == 1
    assert second.revision == 1
    assert first.task_id != second.task_id


def test_cross_process_revision_visible(tmp_path: Path) -> None:
    db_path = tmp_path / "cross.db"
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    saved = store_a.save(_paused_checkpoint())
    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_latest(saved.task_id, _TENANT)
    assert loaded is not None
    assert loaded.revision == saved.revision


def test_store_reopen_preserves_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "reopen.db"
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    first = store.save(_paused_checkpoint())
    second = _save_next(
        store,
        first.model_copy(update={"checkpoint_id": "ckpt_2"}),
        base=first,
    )
    reopened = SQLiteTaskCheckpointStore(db_path=db_path)
    latest = reopened.get_latest(first.task_id, _TENANT)
    assert latest is not None
    assert latest.revision == second.revision


def test_update_without_expected_revision_rejected(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "required.db")
    base = store.save(_paused_checkpoint())
    with pytest.raises(CheckpointRevisionRequiredError) as exc_info:
        store.save(base.model_copy(update={"checkpoint_id": "ckpt_2"}))
    assert exc_info.value.actual_revision == base.revision


def test_validate_not_stale_uses_logical_revision() -> None:
    older = _paused_checkpoint().model_copy(update={"revision": 1})
    newer = older.model_copy(update={"revision": 2, "checkpoint_id": "ckpt_new"})
    result = validate_checkpoint_not_stale(older, newer)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STALE
