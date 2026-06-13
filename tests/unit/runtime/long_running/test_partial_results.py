# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.agent_contract import Agent  # noqa: F401 — stabilize import order
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.partial_results import (
    build_task_progress_view,
    partial_result_from_checkpoint,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_partial_result_from_checkpoint_extracts_runtime_fields(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "partial.db")
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="monitor",
        context=TaskContext(capability="echo.basic"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    LongRunningCoordinator.persist_checkpoint(
        task,
        store,
        progress_message="step 1 complete",
    )
    checkpoint = store.get_latest(task.task_id, "t1")
    assert checkpoint is not None

    partial = partial_result_from_checkpoint(checkpoint)
    assert partial.progress_message == "step 1 complete"
    assert partial.checkpoint_id == checkpoint.checkpoint_id


def test_build_task_progress_view_aggregates_checkpoints() -> None:
    checkpoint = TaskCheckpoint(
        task_id="task_j5",
        tenant_id="t1",
        resume_token="rt_j5",
        task_state=TaskState.WAITING_FOR_HUMAN,
        progress_message="awaiting human input",
        task_snapshot={
            "task_id": "task_j5",
            "tenant_id": "t1",
            "user_id": "u1",
            "message": "hello",
            "context": {"capability": "hitl.basic"},
            "options": {
                "isolation": {},
                "human": {},
                "governance": {},
                "long_running": {"enabled": True},
                "context": {},
            },
            "runtime": {
                "orchestration": {},
                "governance": {"human_request_expires_at": "2026-05-27T12:00:00+00:00"},
                "classification": {},
                "isolation": {},
            },
            "metadata": {},
            "state": "waiting_for_human",
        },
    )
    view = build_task_progress_view(
        task_id="task_j5",
        tenant_id="t1",
        checkpoints=[checkpoint],
    )
    assert view["is_paused"] is True
    assert view["progress_message"] == "awaiting human input"
    assert view["human_request_expires_at"] == "2026-05-27T12:00:00+00:00"
    assert len(view["partial_results"]) == 1
