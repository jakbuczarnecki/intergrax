# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R2-H1 — authoritative resume authority & stale checkpoint closure."""

from __future__ import annotations

import ast
import re
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
    evaluate_checkpoint_resume_eligibility,
    narrow_resume_execution_authority,
    resolve_resume_execution_authority,
    validate_checkpoint_not_stale,
    validate_checkpoint_resume_authority,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-h1"
_FORBIDDEN_AUTHORITY_FALLBACK = re.compile(
    r"restored\.execution_authority",
)
_FORBIDDEN_FRAMEWORK_NAMES = (
    "CheckpointAuthorityResolver",
    "ResumeAuthorityEngine",
    "EnterpriseCheckpointEngine",
    "RecoveryCheckpointRuntime",
    "UniversalResumeManager",
)


def _paused_checkpoint(
    *,
    task_id: str | None = None,
    tenant_id: str = _TENANT,
    created_at_utc: str = "2026-09-09T12:00:00+00:00",
    checkpoint_id: str = "ckpt_h1",
    resume_token: str = "rt-h1",
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
        checkpoint_id=checkpoint_id,
        task_id=resolved_task_id,
        tenant_id=tenant_id,
        resume_token=resume_token,
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc=created_at_utc,
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def _resume_task(
    checkpoint: TaskCheckpoint,
    *,
    execution_authority: ParentExecutionAuthority | None = None,
) -> Task:
    return Task(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        user_id="user",
        message="resume",
        execution_authority=execution_authority,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )


def test_current_authority_none_checkpoint_authority_blocks_resume() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    current = _resume_task(checkpoint, execution_authority=None)
    result = validate_checkpoint_resume_authority(checkpoint, current)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY
    assert (
        evaluate_checkpoint_resume_eligibility(
            checkpoint,
            target_task_id=checkpoint.task_id,
            target_tenant_id=_TENANT,
            latest_checkpoint=checkpoint,
            current_task=current,
        ).eligibility
        is CheckpointResumeEligibility.REJECT_AUTHORITY
    )


def test_coordinator_never_restores_checkpoint_authority_without_current(tmp_path: Path) -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "auth.db")
    store.save(checkpoint)
    task = _resume_task(checkpoint, execution_authority=None)
    with pytest.raises(CheckpointResumeValidationError) as exc_info:
        LongRunningCoordinator.restore_if_resuming(task, store)
    assert exc_info.value.result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY
    assert task.execution_authority is None


def test_current_narrower_than_checkpoint_effective_is_current() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_checkpoint_narrower_than_current_historical_bound_preserved() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_checkpoint_unrestricted_current_restricted_current_wins() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_checkpoint_restricted_current_unrestricted_historical_bound_preserved() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.scoped(("read",))


def test_both_unrestricted_passes() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    effective = resolve_resume_execution_authority(checkpoint, current)
    assert effective == ParentExecutionAuthority.unrestricted_root()


def test_current_authority_unavailable_fail_closed() -> None:
    checkpoint = _paused_checkpoint(
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    result = validate_checkpoint_resume_authority(checkpoint, _resume_task(checkpoint))
    assert result.eligibility is CheckpointResumeEligibility.REJECT_AUTHORITY


def test_malformed_snapshot_authority_fail_closed() -> None:
    checkpoint = _paused_checkpoint()
    checkpoint = checkpoint.model_copy(
        update={"task_snapshot": {"state": "not-a-valid-task"}},
    )
    current = _resume_task(
        checkpoint,
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    result = validate_checkpoint_resume_authority(checkpoint, current)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_MALFORMED
    with pytest.raises(CheckpointResumeValidationError):
        resolve_resume_execution_authority(checkpoint, current)


def test_narrow_helper_is_pure_intersection() -> None:
    current = ParentExecutionAuthority.scoped(("read", "write"))
    historical = ParentExecutionAuthority.scoped(("read", "delete"))
    narrowed = narrow_resume_execution_authority(current, historical)
    assert narrowed == ParentExecutionAuthority.scoped(("read",))


def test_stale_with_earlier_timestamp_blocked() -> None:
    older = _paused_checkpoint(created_at_utc="2026-09-09T10:00:00+00:00")
    newer = _paused_checkpoint(
        task_id=older.task_id,
        checkpoint_id="ckpt_newer",
        created_at_utc="2026-09-09T12:00:00+00:00",
    )
    result = validate_checkpoint_not_stale(older, newer)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STALE


def test_same_timestamp_different_revision_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "same-ts.db")
    older = _paused_checkpoint(
        checkpoint_id="ckpt_old",
        created_at_utc="2026-09-09T12:00:00+00:00",
    )
    older = store.save(older)
    newer = older.model_copy(
        update={
            "checkpoint_id": "ckpt_new",
            "progress_message": "step-2",
            "created_at_utc": "2026-09-09T12:00:00+00:00",
        },
    )
    newer = store.save(newer)
    result = validate_checkpoint_not_stale(older, newer)
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STALE
    assert older.store_sequence is not None
    assert newer.store_sequence is not None
    assert older.store_sequence < newer.store_sequence


def test_missing_timestamp_uses_store_sequence_not_accidental_allow(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "missing-ts.db")
    older = _paused_checkpoint(
        checkpoint_id="ckpt_old",
        created_at_utc="",
    )
    older = store.save(older)
    newer = older.model_copy(
        update={
            "checkpoint_id": "ckpt_new",
            "created_at_utc": "",
            "progress_message": "later",
        },
    )
    newer = store.save(newer)
    result = validate_checkpoint_not_stale(older, store.get_latest(older.task_id, _TENANT))
    assert result.eligibility is CheckpointResumeEligibility.REJECT_STALE


def test_latest_identical_checkpoint_allowed() -> None:
    checkpoint = _paused_checkpoint()
    result = validate_checkpoint_not_stale(checkpoint, checkpoint)
    assert result.eligibility is CheckpointResumeEligibility.ALLOW_RESUME


def test_stale_checkpoint_token_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale-token.db")
    older = store.save(_paused_checkpoint(checkpoint_id="ckpt_old", resume_token="rt-old"))
    newer = store.save(
        _paused_checkpoint(
            task_id=older.task_id,
            checkpoint_id="ckpt_new",
            resume_token="rt-new",
            created_at_utc="2026-09-09T13:00:00+00:00",
        ),
    )
    task = _resume_task(older)
    task.options.long_running.resume_token = older.resume_token
    with pytest.raises(CheckpointResumeValidationError) as exc_info:
        LongRunningCoordinator.restore_if_resuming(task, store)
    assert exc_info.value.result.eligibility is CheckpointResumeEligibility.REJECT_STALE
    latest = store.get_latest(older.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == newer.checkpoint_id


def test_newer_checkpoint_wins(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "newer-wins.db")
    older = store.save(_paused_checkpoint(checkpoint_id="ckpt_old"))
    newer = store.save(
        _paused_checkpoint(
            task_id=older.task_id,
            checkpoint_id="ckpt_new",
            created_at_utc="2026-09-09T14:00:00+00:00",
        ),
    )
    latest = store.get_latest(older.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == newer.checkpoint_id


def test_stale_write_race_rowid_ordering_wins(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale-write.db")
    first = store.save(
        _paused_checkpoint(
            checkpoint_id="ckpt_first",
            created_at_utc="2026-09-09T14:00:00+00:00",
        ),
    )
    second = store.save(
        _paused_checkpoint(
            task_id=first.task_id,
            checkpoint_id="ckpt_second",
            created_at_utc="2026-09-09T10:00:00+00:00",
        ),
    )
    latest = store.get_latest(first.task_id, _TENANT)
    assert latest is not None
    assert latest.checkpoint_id == second.checkpoint_id
    assert second.store_sequence is not None
    assert first.store_sequence is not None
    assert second.store_sequence > first.store_sequence


def test_coordinator_source_has_no_checkpoint_authority_fallback() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "coordinator.py"
    ).read_text(encoding="utf-8-sig")
    assert _FORBIDDEN_AUTHORITY_FALLBACK.search(source) is None


def test_ast_no_direct_checkpoint_authority_assignment() -> None:
    coordinator_path = _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "coordinator.py"
    tree = ast.parse(coordinator_path.read_text(encoding="utf-8-sig"))
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if (
                isinstance(target.value, ast.Name)
                and target.value.id == "restored"
                and target.attr == "execution_authority"
            ):
                forbidden.append(ast.get_source_segment(
                    coordinator_path.read_text(encoding="utf-8-sig"),
                    node,
                ) or "restored.execution_authority assignment")
    assert forbidden == []


def test_no_new_authority_engine() -> None:
    long_running = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
    for path in long_running.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_FRAMEWORK_NAMES:
            assert name not in text
