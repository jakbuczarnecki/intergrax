# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient
from intergrax.debug.app import create_debug_app
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.events.trace_bridge import (
    trace_bridge_subject_from_tags,
    trace_event_to_runtime_event,
)
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

pytestmark = pytest.mark.unit

_RUN_ID = "run_0123456789abcdef0123456789abcdef"
_TASK_ID = "task_0123456789abcdef0123456789abcdef"
_BRIDGE_ATTEMPT_ID = "attempt_0123456789abcdef0123456789abcdef"
_BRIDGE_EXECUTION_ID = "exec_0123456789abcdef0123456789abcdef"


def _seed_store(db_path) -> TraceEvent:
    store = SQLiteRunTraceStore(db_path=db_path)
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id=_RUN_ID,
        seq=1,
        ts_utc="2026-05-27T10:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> completed",
        tags={
            "task_id": _TASK_ID,
            "task_state": "completed",
            "capability": "echo.basic",
        },
    )
    store.append_event(event)
    store.finalize_run(
        _RUN_ID,
        RunMetadata(
            run_id=_RUN_ID,
            session_id="s1",
            user_id="u1",
            tenant_id="t1",
            started_at_utc="2026-05-27T10:00:00+00:00",
            stats=RunStats(duration_ms=42, llm_usage={}),
        ),
    )
    return event


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "debug_api.db"
    trace_event = _seed_store(db_path)
    runtime_event_store = InMemoryRuntimeEventStore()
    bridged = trace_event_to_runtime_event(
        trace_event,
        trace_bridge_subject_from_tags(
            tenant_id="t1",
            task_id=_TASK_ID,
        ),
        run_id=_RUN_ID,
        attempt_id=_BRIDGE_ATTEMPT_ID,
        execution_id=_BRIDGE_EXECUTION_ID,
    )
    runtime_event_store.append(bridged, tenant_id="t1")
    app = create_debug_app(db_path=db_path, runtime_event_store=runtime_event_store)
    with TestClient(app) as test_client:
        yield test_client


def test_debug_api_list_tasks(client: TestClient):
    response = client.get("/debug/tasks", params={"tenant": "t1", "limit": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["tenant_id"] == "t1"
    assert payload["count"] == 1
    assert payload["runs"][0]["run_id"] == "run_0123456789abcdef0123456789abcdef"


def test_debug_api_show_task(client: TestClient):
    response = client.get("/debug/tasks/run_0123456789abcdef0123456789abcdef", params={"tenant": "t1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run_0123456789abcdef0123456789abcdef"
    assert payload["stats"]["duration_ms"] == 42
    assert payload["event_count"] == 1


def test_debug_api_trace_with_runtime(client: TestClient):
    response = client.get(
        "/debug/tasks/run_0123456789abcdef0123456789abcdef/trace",
        params={"tenant": "t1", "include_runtime": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run_0123456789abcdef0123456789abcdef"
    assert len(payload["trace_events"]) == 1
    assert len(payload["runtime_events"]) == 1


def test_debug_api_task_not_found(client: TestClient):
    response = client.get("/debug/tasks/missing-run", params={"tenant": "t1"})
    assert response.status_code == 404


def test_debug_api_missing_db():
    app = create_debug_app(db_path="nonexistent/path/trace.db")
    with TestClient(app) as client:
        response = client.get("/debug/tasks", params={"tenant": "t1"})
    assert response.status_code == 503


@pytest.mark.gate
@pytest.mark.no_ci
def test_debug_api_uses_injected_trace_store_without_sqlite():
    from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore

    store = InMemoryRunTraceStore()
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id="run_fedcba9876543210fedcba9876543210",
        seq=1,
        ts_utc="2026-05-27T10:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> completed",
        tags={"task_id": "run_fedcba9876543210fedcba9876543210", "task_state": "completed"},
    )
    store.append_event(event)
    store.finalize_run(
        "run_fedcba9876543210fedcba9876543210",
        RunMetadata(
            run_id="run_fedcba9876543210fedcba9876543210",
            session_id="s1",
            user_id="u1",
            tenant_id="t1",
            started_at_utc="2026-05-27T10:00:00+00:00",
            stats=RunStats(duration_ms=42, llm_usage={}),
        ),
    )
    app = create_debug_app(trace_store=store)
    with TestClient(app) as client:
        response = client.get("/debug/tasks", params={"tenant": "t1", "limit": 10})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["runs"][0]["run_id"] == "run_fedcba9876543210fedcba9876543210"
