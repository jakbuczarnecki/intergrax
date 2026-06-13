# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.debug.app import create_debug_app
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel

pytestmark = pytest.mark.unit


def _seed_store(db_path) -> SQLiteRunTraceStore:
    store = SQLiteRunTraceStore(db_path=db_path)
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id="run-debug-api-1",
        seq=1,
        ts_utc="2026-05-27T10:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> completed",
        tags={
            "task_id": "run-debug-api-1",
            "task_state": "completed",
            "capability": "echo.basic",
        },
    )
    store.append_event(event)
    store.finalize_run(
        "run-debug-api-1",
        RunMetadata(
            run_id="run-debug-api-1",
            session_id="s1",
            user_id="u1",
            tenant_id="t1",
            started_at_utc="2026-05-27T10:00:00+00:00",
            stats=RunStats(duration_ms=42, llm_usage={}),
        ),
    )
    return store


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "debug_api.db"
    _seed_store(db_path)
    app = create_debug_app(db_path=db_path)
    with TestClient(app) as test_client:
        yield test_client


def test_debug_api_list_tasks(client: TestClient):
    response = client.get("/debug/tasks", params={"tenant": "t1", "limit": 10})
    assert response.status_code == 200
    payload = response.json()
    assert payload["tenant_id"] == "t1"
    assert payload["count"] == 1
    assert payload["runs"][0]["run_id"] == "run-debug-api-1"


def test_debug_api_show_task(client: TestClient):
    response = client.get("/debug/tasks/run-debug-api-1", params={"tenant": "t1"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-debug-api-1"
    assert payload["stats"]["duration_ms"] == 42
    assert payload["event_count"] == 1


def test_debug_api_trace_with_runtime(client: TestClient):
    response = client.get(
        "/debug/tasks/run-debug-api-1/trace",
        params={"tenant": "t1", "include_runtime": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "run-debug-api-1"
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
        run_id="run-debug-api-mem",
        seq=1,
        ts_utc="2026-05-27T10:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> completed",
        tags={"task_id": "run-debug-api-mem", "task_state": "completed"},
    )
    store.append_event(event)
    store.finalize_run(
        "run-debug-api-mem",
        RunMetadata(
            run_id="run-debug-api-mem",
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
    assert response.json()["runs"][0]["run_id"] == "run-debug-api-mem"
