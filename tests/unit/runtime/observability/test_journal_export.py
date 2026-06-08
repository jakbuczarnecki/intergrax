# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import JOURNAL_SCHEMA_VERSION
from intergrax.runtime.nexus.tracing.persistence_models import (
    PersistedRun,
    RunMetadata,
    RunStats,
    SerializedTraceEvent,
)
from intergrax.runtime.observability.journal_export import (
    build_journal_export_snapshot,
    build_journal_ref_payload,
    count_parser_traces_in_trace_events,
    render_journal_otlp_json,
)

pytestmark = pytest.mark.gate


def _persisted_run(*events: SerializedTraceEvent) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id="run-export-1",
            session_id="s1",
            user_id="u1",
            tenant_id="tenant-a",
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=list(events),
    )


def test_journal_ref_payload_includes_event_and_parser_counts() -> None:
    trace = SerializedTraceEvent(
        event_id="trace-parser",
        run_id="run-export-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="pipeline",
        step="document_ingest",
        message="parsed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={
            "integration_parser_trace": {"parser_id": "pdf", "attempts": []},
            "source": "nexus_run_trace",
        },
        artifact_refs=[],
    )
    lifecycle = SerializedTraceEvent(
        event_id="trace-done",
        run_id="run-export-1",
        seq=2,
        ts_utc="2026-06-07T10:00:02+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> completed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": "run-export-1", "task_state": "completed"},
        artifact_refs=[],
    )
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id="evt_bus_1",
            tenant_id="tenant-a",
            task_id="run-export-1",
            run_id="run-export-1",
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            severity=EventSeverity.INFO,
            payload={"policy_action": "allow"},
            timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
            correlation_id="run-export-1",
        ),
        tenant_id="tenant-a",
    )

    ref = build_journal_ref_payload(
        _persisted_run(trace, lifecycle),
        runtime_store=store,
    )
    assert ref is not None
    assert ref["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert ref["run_id"] == "run-export-1"
    assert ref["tenant_id"] == "tenant-a"
    assert ref["event_count"] == 3
    assert ref["parser_trace_count"] == 1


def test_journal_export_snapshot_serializes_unified_journal() -> None:
    trace = SerializedTraceEvent(
        event_id="trace-1",
        run_id="run-export-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> completed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": "run-export-1", "task_state": "completed"},
        artifact_refs=[],
    )
    snapshot = build_journal_export_snapshot(_persisted_run(trace))
    assert snapshot.event_count == 1
    assert snapshot.events[0]["event_type"] == RuntimeEventType.TASK_COMPLETED.value
    assert snapshot.parser_trace_count == 0


def test_render_journal_otlp_json_produces_resource_spans() -> None:
    trace = SerializedTraceEvent(
        event_id="trace-otlp",
        run_id="run-export-1",
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> completed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": "run-export-1", "task_state": "completed"},
        artifact_refs=[],
    )
    snapshot = build_journal_export_snapshot(_persisted_run(trace))
    otlp = render_journal_otlp_json(snapshot)
    resource_spans = otlp["resourceSpans"]
    assert len(resource_spans) == 1
    spans = resource_spans[0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == RuntimeEventType.TASK_COMPLETED.value
    assert len(spans[0]["traceId"]) == 32
    assert len(spans[0]["spanId"]) == 16


def test_count_parser_traces_in_trace_events() -> None:
    events = [
        SerializedTraceEvent(
            event_id="a",
            run_id="r",
            seq=1,
            ts_utc="t",
            level="info",
            component="pipeline",
            step="x",
            message="m",
            payload_schema_id=None,
            payload_schema_version=None,
            payload=None,
            tags={"integration_parser_trace": {"parser_id": "x"}},
            artifact_refs=[],
        ),
        SerializedTraceEvent(
            event_id="b",
            run_id="r",
            seq=2,
            ts_utc="t",
            level="info",
            component="pipeline",
            step="y",
            message="m",
            payload_schema_id=None,
            payload_schema_version=None,
            payload=None,
            tags={},
            artifact_refs=[],
        ),
    ]
    assert count_parser_traces_in_trace_events(events) == 1
