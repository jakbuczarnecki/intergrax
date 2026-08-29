# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_event_id, mint_execution_id, mint_run_id, mint_task_id
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
    build_journal_ref,
    build_journal_ref_payload,
    count_parser_traces_in_trace_events,
    render_journal_otlp_json,
)

pytestmark = pytest.mark.gate

_TENANT = "tenant-a"


def _persisted_run(*events: SerializedTraceEvent, run_id: str) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=_TENANT,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
        events=list(events),
    )


def _runtime_event(
    *,
    task_id: str,
    run_id: str,
    event_type: RuntimeEventType,
    event_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        payload=dict(payload or {}),
        timestamp=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        correlation_id=run_id,
    )


def test_journal_ref_payload_includes_event_and_parser_counts() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    trace = SerializedTraceEvent(
        event_id="trace-parser",
        run_id=run_id,
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
        run_id=run_id,
        seq=2,
        ts_utc="2026-06-07T10:00:02+00:00",
        level="info",
        component="planner",
        step="task_lifecycle",
        message="task state -> completed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"task_id": task_id, "task_state": "completed"},
        artifact_refs=[],
    )
    store = InMemoryRuntimeEventStore()
    store.append(
        _runtime_event(
            task_id=task_id,
            run_id=run_id,
            event_type=RuntimeEventType.POLICY_DECISION,
            event_id=mint_event_id(),
            payload={"policy_action": "allow"},
        ),
        tenant_id=_TENANT,
    )

    ref = build_journal_ref_payload(
        _persisted_run(trace, lifecycle, run_id=run_id),
        runtime_store=store,
    )
    assert ref is not None
    assert ref["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert ref["run_id"] == run_id
    assert ref["tenant_id"] == _TENANT
    assert ref["event_count"] == 1
    assert ref["parser_trace_count"] == 1


def test_journal_export_snapshot_serializes_unified_journal() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _runtime_event(
            task_id=task_id,
            run_id=run_id,
            event_type=RuntimeEventType.TASK_COMPLETED,
        ),
        tenant_id=_TENANT,
    )
    snapshot = build_journal_export_snapshot(
        _persisted_run(run_id=run_id),
        runtime_store=store,
    )
    assert snapshot.event_count == 1
    assert snapshot.events[0]["event_type"] == RuntimeEventType.TASK_COMPLETED.value
    assert snapshot.parser_trace_count == 0


def test_render_journal_otlp_json_produces_resource_spans() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _runtime_event(
            task_id=task_id,
            run_id=run_id,
            event_type=RuntimeEventType.TASK_COMPLETED,
        ),
        tenant_id=_TENANT,
    )
    snapshot = build_journal_export_snapshot(
        _persisted_run(run_id=run_id),
        runtime_store=store,
    )
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


class _RecordingStore(InMemoryRuntimeEventStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_positioned_for_run_calls: list[tuple[str, str, int]] = []

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through=None,
    ):
        self.list_positioned_for_run_calls.append((run_id, tenant_id, limit))
        return super().list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
        )


def test_journal_ref_empty_canonical_store_keeps_parser_trace_count() -> None:
    run_id = mint_run_id()
    parser_trace = SerializedTraceEvent(
        event_id="trace-parser",
        run_id=run_id,
        seq=1,
        ts_utc="2026-06-07T10:00:01+00:00",
        level="info",
        component="pipeline",
        step="document_ingest",
        message="parsed",
        payload_schema_id=None,
        payload_schema_version=None,
        payload=None,
        tags={"integration_parser_trace": {"parser_id": "pdf", "attempts": []}},
        artifact_refs=[],
    )
    store = _RecordingStore()
    persisted = _persisted_run(parser_trace, run_id=run_id)

    ref = build_journal_ref(persisted, runtime_store=store)

    assert ref is not None
    assert ref.event_count == 0
    assert ref.parser_trace_count == 1
    assert store.list_positioned_for_run_calls == [(run_id, _TENANT, 2000)]


def test_journal_export_snapshot_requires_actual_runtime_store() -> None:
    run_id = mint_run_id()
    store = _RecordingStore()
    snapshot = build_journal_export_snapshot(_persisted_run(run_id=run_id), runtime_store=store)
    assert snapshot.event_count == 0
    assert snapshot.events == []
    assert snapshot.parser_trace_count == 0
    assert store.list_positioned_for_run_calls == [(run_id, _TENANT, 2000)]
