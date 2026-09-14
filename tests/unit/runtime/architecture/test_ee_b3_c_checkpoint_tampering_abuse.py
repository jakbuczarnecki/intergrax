# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-09 checkpoint tampering abuse."""

from __future__ import annotations

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
    validate_checkpoint_identity_binding,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    minimal_runtime_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskLongRunningOptions,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-b3c-ckpt"


def _paused_checkpoint() -> TaskCheckpoint:
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-b3c"),
        ),
    )
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=_TENANT,
        resume_token="rt-b3c",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        created_at_utc="2026-09-14T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def test_ee_b3_c_tampered_run_id_rejected_on_resume_binding() -> None:
    checkpoint = _paused_checkpoint()
    forged_run = mint_run_id()
    result = validate_checkpoint_identity_binding(
        checkpoint,
        target_task_id=checkpoint.task_id,
        target_tenant_id=_TENANT,
        target_run_id=forged_run,
    )
    assert result.eligibility is CheckpointResumeEligibility.REJECT_IDENTITY


def test_ee_b3_c_stale_revision_write_rejected(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "b3c.db")
    first = store.save(_paused_checkpoint())
    store.save(
        first.model_copy(
            update={"checkpoint_id": "ckpt_v2", "progress_message": "v2"},
        ),
        expected_revision=first.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            first.model_copy(
                update={"checkpoint_id": "ckpt_stale", "progress_message": "tampered"},
            ),
            expected_revision=first.revision,
        )


def test_ee_b3_c_runtime_validate_canonical_rejects_tree_mismatch() -> None:
    checkpoint = _paused_checkpoint()
    runtime = checkpoint.runtime
    assert runtime is not None
    other_attempt = mint_attempt_id()
    tampered = runtime.model_copy(update={"attempt_id": other_attempt})
    with pytest.raises(ValueError, match="attempt_id mismatch"):
        tampered.validate_canonical()
