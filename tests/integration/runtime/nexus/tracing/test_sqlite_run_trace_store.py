
import pytest
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import (
    SQLiteRunTraceStore,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent, TraceLevel
from intergrax.runtime.nexus.tracing.persistence_models import (
    RunMetadata,
    RunStats,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from testing_support.builder import prepare_sqlite_db

pytestmark = pytest.mark.integrations


def test_sqlite_run_trace_store_persist_and_read():    

    db_path = prepare_sqlite_db("sqlite_trace.db")

    store = SQLiteRunTraceStore(db_path=db_path)

    # ---- Create event ----
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id="run-1",
        seq=1,
        ts_utc="2026-01-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="test_step",
        message="hello",
        payload=None,
        tags={},
        artifact_refs=(),
    )

    store.append_event(event)

    # ---- Create metadata ----
    stats = RunStats(
        duration_ms=10,
        llm_usage={},
    )

    metadata = RunMetadata(
        run_id="run-1",
        session_id="s1",
        user_id="u1",
        tenant_id="t1",
        started_at_utc="2026-01-01T00:00:00Z",
        stats=stats,
        error=None,
    )

    store.finalize_run("run-1", metadata)

    # ---- Read ----
    persisted = store.read_run("run-1", metadata.tenant_id)

    assert persisted.metadata.run_id == "run-1"
    assert persisted.metadata.session_id == "s1"
    assert persisted.metadata.tenant_id == "t1"
    assert persisted.metadata.stats.duration_ms == 10

    assert len(persisted.events) == 1
    assert persisted.events[0]["run_id"] == "run-1"
