# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.governance.in_memory_metrics_store import InMemoryMetricsStore
from intergrax.runtime.metrics.export import export_run_metrics, persist_run_metrics
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunStats,
    SerializedTraceEvent,
)

pytestmark = pytest.mark.gate


def _persisted_run() -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id="run_1",
            session_id="sess",
            user_id="u1",
            tenant_id="lab",
            started_at_utc="2026-05-27T00:00:00Z",
            stats=RunStats(
                duration_ms=120,
                llm_usage={"cost": 0.01, "total_tokens": 42},
            ),
        ),
        events=[
            SerializedTraceEvent(
                event_id="e1",
                run_id="run_1",
                seq=1,
                ts_utc="2026-05-27T00:00:00Z",
                level="info",
                component="nexus",
                step="graph",
                message="graph node start: n1",
                payload_schema_id="llm.usage",
                payload_schema_version=1,
                payload={},
                tags={},
                artifact_refs=[],
            )
        ],
    )


def test_export_run_metrics_includes_trace_summary():
    exported = export_run_metrics(_persisted_run(), agent_id="echo")
    assert exported.run_id == "run_1"
    assert exported.duration_ms == 120
    assert exported.total_tokens == 42
    assert exported.trace_summary["graph_events"] == 1
    assert exported.trace_summary["llm_events"] == 1


def test_persist_run_metrics_saves_synthetic_behavioral_from_trace():
    store = InMemoryMetricsStore()
    exported = persist_run_metrics(
        store=store,
        persisted=_persisted_run(),
        agent_id="echo",
    )
    assert exported.agent_id == "echo"
    assert exported.behavioral is not None
    assert len(store.get_recent("echo", limit=5)) == 1
