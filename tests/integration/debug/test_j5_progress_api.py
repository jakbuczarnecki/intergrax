# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.agents.agent_contract import Agent  # noqa: F401 — stabilize import order
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.debug.app import create_debug_app
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.fixture
def progress_client(tmp_path):
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "progress.db")
    event_store = InMemoryRuntimeEventStore()
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="long monitor",
        context=TaskContext(capability="hitl.basic"),
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    LongRunningCoordinator.persist_checkpoint(
        task,
        checkpoint_store,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        progress_message="awaiting human input",
    )
    event_store.append(
        RuntimeEvent(
            tenant_id="t1",
            task_id=task.task_id,
            run_id=task.task_id,
            event_type=RuntimeEventType.TASK_PROGRESS,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload={"progress_message": "awaiting human input"},
        ),
        tenant_id="t1",
    )
    app = create_debug_app(
        checkpoint_store=checkpoint_store,
        runtime_event_store=event_store,
    )
    with TestClient(app) as client:
        yield client, task


def test_debug_api_task_progress(progress_client) -> None:
    client, task = progress_client
    response = client.get(
        f"/debug/tasks/{task.task_id}/progress",
        params={"tenant": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task.task_id
    assert payload["is_paused"] is True
    assert payload["progress_message"] == "awaiting human input"
    assert payload["checkpoint_count"] == 1
    assert payload["progress_event_count"] == 1
    assert len(payload["partial_results"]) == 1
    assert payload["latest_partial_result"]["checkpoint_id"]


def test_debug_api_task_progress_not_found(tmp_path) -> None:
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "empty.db")
    app = create_debug_app(checkpoint_store=checkpoint_store)
    with TestClient(app) as client:
        response = client.get("/debug/tasks/missing/progress", params={"tenant": "t1"})
    assert response.status_code == 404
