# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions


@pytest.mark.unit
@pytest.mark.gate
def test_checkpoint_store_roundtrip(tmp_path):
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="monitor vendors",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
        ),
    )
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        task,
        store,
        progress_message="step 1 complete",
    )
    assert checkpoint.resume_token
    loaded = store.get_by_token(task.task_id, "t1", checkpoint.resume_token)
    assert loaded is not None
    assert loaded.progress_message == "step 1 complete"


@pytest.mark.unit
@pytest.mark.gate
def test_restore_if_resuming_merges_snapshot(tmp_path):
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    original = Task(
        tenant_id="t1",
        user_id="u1",
        message="paused work",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    checkpoint = LongRunningCoordinator.persist_checkpoint(
        original,
        store,
        progress_message="awaiting human input",
    )

    resume_task = Task(
        tenant_id="t1",
        user_id="u1",
        task_id=original.task_id,
        message="ignored until restore",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=checkpoint.resume_token,
            ),
        ),
    )
    restored = LongRunningCoordinator.restore_if_resuming(resume_task, store)
    assert restored is not None
    assert resume_task.message == "paused work"
    assert resume_task.runtime.orchestration.checkpoint_id == checkpoint.checkpoint_id
