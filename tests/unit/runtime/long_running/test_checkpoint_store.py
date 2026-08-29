# © Artur Czarnecki. All rights reserved.

import pytest

pytestmark = pytest.mark.no_ci

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
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
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        checkpoint = LongRunningCoordinator.persist_checkpoint(
            task,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_message="step 1 complete",
        )
    finally:
        reset_active_execution_identity(token)
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
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        checkpoint = LongRunningCoordinator.persist_checkpoint(
            original,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_message="awaiting human input",
        )
    finally:
        reset_active_execution_identity(token)

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
