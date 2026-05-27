# © Artur Czarnecki. All rights reserved.

import json

import pytest

from intergrax.debug.cli import main
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from testing_support.builder import prepare_sqlite_db

pytestmark = pytest.mark.unit


def _seed_store(db_path) -> SQLiteRunTraceStore:
    store = SQLiteRunTraceStore(db_path=db_path)
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id="run-debug-1",
        seq=1,
        ts_utc="2026-05-27T10:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> completed",
        tags={
            "task_id": "run-debug-1",
            "task_state": "completed",
            "capability": "echo.basic",
        },
    )
    store.append_event(event)
    store.finalize_run(
        "run-debug-1",
        RunMetadata(
            run_id="run-debug-1",
            session_id="s1",
            user_id="u1",
            tenant_id="t1",
            started_at_utc="2026-05-27T10:00:00+00:00",
            stats=RunStats(duration_ms=42, llm_usage={}),
        ),
    )
    return store


def test_sqlite_list_runs():
    db_path = prepare_sqlite_db("debug_list_runs.db")
    _seed_store(db_path)
    store = SQLiteRunTraceStore(db_path=db_path)
    runs = store.list_runs("t1", limit=10)
    assert len(runs) == 1
    assert runs[0].run_id == "run-debug-1"
    assert runs[0].event_count == 1


def test_debug_cli_list_show_trace(capsys):
    db_path = prepare_sqlite_db("debug_cli.db")
    _seed_store(db_path)

    assert main(["--db", str(db_path), "tasks", "list", "--tenant", "t1"]) == 0
    out = capsys.readouterr().out
    assert "run-debug-1" in out

    assert main(["--db", str(db_path), "tasks", "show", "run-debug-1", "--tenant", "t1"]) == 0
    out = capsys.readouterr().out
    assert "duration_ms: 42" in out

    assert (
        main(
            [
                "--db",
                str(db_path),
                "tasks",
                "trace",
                "run-debug-1",
                "--tenant",
                "t1",
                "--format",
                "json",
                "--runtime",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-debug-1"
    assert len(payload["trace_events"]) == 1
    assert len(payload["runtime_events"]) == 1


def test_debug_cli_missing_db():
    assert main(["--db", "nonexistent/path/trace.db", "tasks", "list"]) == 1
