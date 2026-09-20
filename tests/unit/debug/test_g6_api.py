# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.debug.app import create_debug_app
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)
from testing_support.builder import canonical_governed_execution_scope
from testing_support.nexus_hitl_test_agent import NexusBasicHitlTestAgent
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = pytest.mark.unit

_HitlAgent = NexusBasicHitlTestAgent


@pytest.fixture
def event_store():
    task_id = mint_task_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            tenant_id="t1",
            task_id=task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
            event_type=RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
            phase=ExecutionPhase.HUMAN_APPROVAL,
            payload={"human_request": {"urgency": "high"}},
        ),
        tenant_id="t1",
    )
    store.fixture_task_id = task_id
    return store


@pytest.fixture
def checkpoint_store(tmp_path):
    return SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")


@pytest.fixture
def events_client(event_store):
    app = create_debug_app(runtime_event_store=event_store)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def checkpoints_client(checkpoint_store):
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="paused",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.state = TaskState.WAITING_FOR_HUMAN
    with canonical_governed_execution_scope("g6-debug-checkpoints"):
        LongRunningCoordinator.persist_checkpoint(
            task,
            checkpoint_store,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            progress_message="awaiting human input",
        )
    app = create_debug_app(checkpoint_store=checkpoint_store)
    with TestClient(app) as client:
        yield client, task


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_runtime_events(events_client: TestClient, event_store):
    response = events_client.get(
        f"/debug/tasks/{event_store.fixture_task_id}/events",
        params={"tenant": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["events"][0]["event_type"] == "human_approval_requested"


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_runtime_events_not_configured():
    app = create_debug_app()
    with TestClient(app) as client:
        response = client.get("/debug/tasks/x/events", params={"tenant": "t1"})
    assert response.status_code == 503


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_checkpoints(checkpoints_client):
    client, task = checkpoints_client
    response = client.get(
        f"/debug/tasks/{task.task_id}/checkpoints",
        params={"tenant": "t1"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["checkpoints"][0]["resume_token"]


@pytest.mark.asyncio
@pytest.mark.gate
@pytest.mark.no_ci
async def test_debug_api_human_response_resume(tmp_path):
    registry = AgentRegistry()
    registry.register(_HitlAgent())
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    event_store = InMemoryRuntimeEventStore()

    loop = NexusLoop(
        registry,
        checkpoint_store=checkpoint_store,
        runtime_event_store=event_store,
    )
    runner = UnifiedTaskRunner(
        loop,
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    paused = await runner.run_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="sensitive action",
            context=TaskContext(capability="hitl.basic"),
            options=TaskExecutionOptions(
                long_running=TaskLongRunningOptions(enabled=True),
            ),
        ),
        run_id=mint_run_id(),
    )
    assert paused.state == TaskState.WAITING_FOR_HUMAN
    assert checkpoint_store.get_latest(paused.task_id, "t1") is not None

    app = create_debug_app(
        registry=registry,
        checkpoint_store=checkpoint_store,
        runtime_event_store=event_store,
    )
    with TestClient(app) as client:
        approved = client.post(
            f"/debug/tasks/{paused.task_id}/human-response",
            params={"tenant": "t1"},
            json={"response": "approve"},
        )
        assert approved.status_code == 200
        body = approved.json()
        assert body["state"] == "completed"

        not_configured = create_debug_app()
        with TestClient(not_configured) as bare_client:
            denied = bare_client.post(
                f"/debug/tasks/{paused.task_id}/human-response",
                params={"tenant": "t1"},
                json={"response": "approve"},
            )
            assert denied.status_code == 503
