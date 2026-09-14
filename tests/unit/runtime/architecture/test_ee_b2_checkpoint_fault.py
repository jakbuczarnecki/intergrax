# © Artur Czarnecki. All rights reserved.

"""EE-B2 — checkpoint persistence fault injection."""

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

_TENANT = "tenant-ee-b2-ckpt"


def _paused_checkpoint() -> TaskCheckpoint:
    task_id = str(mint_task_id())
    task = Task(
        task_id=task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-b2"),
        ),
    )
    return TaskCheckpoint(
        checkpoint_id=f"ckpt_{mint_task_id()}",
        task_id=task_id,
        tenant_id=_TENANT,
        resume_token="rt-b2",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="paused",
        created_at_utc="2026-09-13T12:00:00+00:00",
        runtime=minimal_runtime_checkpoint(
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )


def test_ee_b2_stale_checkpoint_write_rejected_no_corrupt_resume(
    tmp_path: Path,
) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    first = store.save(_paused_checkpoint())
    store.save(
        first.model_copy(update={"checkpoint_id": "ckpt_v2", "progress_message": "v2"}),
        expected_revision=first.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            first.model_copy(
                update={"checkpoint_id": "ckpt_stale", "progress_message": "stale"}
            ),
            expected_revision=first.revision,
        )
