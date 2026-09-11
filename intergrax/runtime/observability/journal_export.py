# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded safe run-journal export — typed envelopes and OTLP-style snapshots (OBS-BUS-6, R3)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.w3c_trace_context import (
    is_valid_traceparent,
    parse_traceparent,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.unified_run_journal import (
    JOURNAL_SCHEMA_VERSION,
    read_run_journal_page,
)
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun
if TYPE_CHECKING:
    from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope

JOURNAL_EXPORT_SCHEMA_VERSION = "journal_export.v2"


@dataclass(frozen=True, slots=True)
class JournalRef:
    """Lightweight pointer attached to ``TASK_COMPLETED`` payloads."""

    schema_version: str
    run_id: str
    tenant_id: str
    event_count: int
    parser_trace_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "event_count": self.event_count,
            "parser_trace_count": self.parser_trace_count,
        }


@dataclass(frozen=True, slots=True)
class JournalExportSnapshot:
    """Bounded, content-safe export snapshot for OTLP dual-write and operator tooling."""

    schema_version: str
    journal_schema_version: str
    run_id: str
    tenant_id: str
    event_count: int
    parser_trace_count: int
    events: tuple[ObservabilityExportEnvelope, ...]
    is_complete: bool
    has_continuation: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "journal_schema_version": self.journal_schema_version,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "event_count": self.event_count,
            "parser_trace_count": self.parser_trace_count,
            "is_complete": self.is_complete,
            "has_continuation": self.has_continuation,
            "events": [envelope.model_dump(mode="json") for envelope in self.events],
        }


def build_journal_ref(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence,
    limit: int = 2000,
) -> JournalRef | None:
    """Build a lightweight journal reference for terminal runtime events."""
    page = read_run_journal_page(
        runtime_store,
        tenant_id=persisted.metadata.tenant_id,
        run_id=persisted.metadata.run_id,
        page_size=limit,
    )
    journal = list(page.events)
    if not journal and not persisted.events:
        return None
    return JournalRef(
        schema_version=JOURNAL_SCHEMA_VERSION,
        run_id=persisted.metadata.run_id,
        tenant_id=persisted.metadata.tenant_id,
        event_count=len(journal),
        parser_trace_count=count_parser_traces_in_trace_events(persisted.events),
    )


def build_journal_ref_payload(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence,
    limit: int = 2000,
) -> Dict[str, Any] | None:
    """``TASK_COMPLETED`` payload fragment with unified journal metadata."""
    ref = build_journal_ref(persisted, runtime_store=runtime_store, limit=limit)
    if ref is None:
        return None
    return ref.to_dict()


def build_journal_export_snapshot(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence,
    limit: int = 2000,
) -> JournalExportSnapshot:
    """Serialize one bounded journal page as typed safe export envelopes."""
    page = read_run_journal_page(
        runtime_store,
        tenant_id=persisted.metadata.tenant_id,
        run_id=persisted.metadata.run_id,
        page_size=limit,
    )
    journal = list(page.events)
    envelopes = tuple(serialize_runtime_event(event) for event in journal)
    return JournalExportSnapshot(
        schema_version=JOURNAL_EXPORT_SCHEMA_VERSION,
        journal_schema_version=JOURNAL_SCHEMA_VERSION,
        run_id=persisted.metadata.run_id,
        tenant_id=persisted.metadata.tenant_id,
        event_count=len(journal),
        parser_trace_count=count_parser_traces_in_trace_events(persisted.events),
        events=envelopes,
        is_complete=page.is_complete,
        has_continuation=page.next_cursor is not None,
    )


def serialize_runtime_event(event: RuntimeEvent) -> ObservabilityExportEnvelope:
    """Project a runtime event into the canonical observability export envelope."""
    from intergrax.runtime.observability.export_boundary import (
        envelope_from_runtime_event,
        envelope_is_content_safe,
    )

    envelope = envelope_from_runtime_event(event)
    if not envelope_is_content_safe(envelope):
        raise ValueError("runtime event export envelope failed content-safety validation")
    return envelope


def count_parser_traces_in_trace_events(events: Sequence[Any]) -> int:
    """Count persisted trace rows carrying ``integration_parser_trace`` tags."""
    count = 0
    for event in events:
        tags = _trace_row_tags(event)
        trace = tags.get("integration_parser_trace")
        if isinstance(trace, dict):
            count += 1
    return count


def render_journal_otlp_json(snapshot: JournalExportSnapshot) -> Dict[str, Any]:
    """
    OTLP-inspired JSON trace snapshot for observability backends / debug export.

    Not a full OTLP protobuf encoder — stable JSON for log sinks and HTTP routes.
    Input must be a bounded safe ``JournalExportSnapshot`` (no raw runtime events).
    """
    run_id = snapshot.run_id
    tenant_id = snapshot.tenant_id
    spans: List[Dict[str, Any]] = []
    for envelope in snapshot.events:
        event_id = envelope.event_id
        event_type = envelope.event_type or "unknown"
        traceparent_raw = envelope.w3c_traceparent
        if traceparent_raw and is_valid_traceparent(traceparent_raw):
            parsed = parse_traceparent(traceparent_raw)
            trace_id = parsed.trace_id
            span_id = parsed.parent_id
        else:
            trace_id = _otlp_hex_id(run_id, length=32)
            span_id = _otlp_hex_id(event_id, length=16)
        span: Dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": event_type,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": _timestamp_to_unix_nano(envelope.recorded_at),
            "attributes": _span_attributes_from_envelope(envelope, tenant_id=tenant_id),
        }
        parent_event_id = envelope.parent_event_id
        if parent_event_id.strip():
            span["parentSpanId"] = _otlp_hex_id(parent_event_id, length=16)
        spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "intergrax.harness"}},
                        {"key": "tenant.id", "value": {"stringValue": tenant_id}},
                        {"key": "run.id", "value": {"stringValue": run_id}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "intergrax.unified_run_journal"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _span_attributes_from_envelope(
    envelope: ObservabilityExportEnvelope,  # noqa: F821 — TYPE_CHECKING
    *,
    tenant_id: str,
) -> List[Dict[str, Any]]:
    attrs: List[Dict[str, Any]] = [
        {"key": "intergrax.event_id", "value": {"stringValue": envelope.event_id}},
        {"key": "intergrax.tenant_id", "value": {"stringValue": tenant_id}},
        {"key": "intergrax.task_id", "value": {"stringValue": envelope.task_id}},
        {"key": "intergrax.phase", "value": {"stringValue": envelope.execution_phase}},
    ]
    if envelope.agent_id:
        attrs.append({"key": "intergrax.agent_id", "value": {"stringValue": envelope.agent_id}})
    if envelope.parent_event_id:
        attrs.append(
            {"key": "intergrax.parent_event_id", "value": {"stringValue": envelope.parent_event_id}}
        )
    if envelope.w3c_traceparent:
        attrs.append({"key": "w3c.traceparent", "value": {"stringValue": envelope.w3c_traceparent}})
    if envelope.w3c_tracestate:
        attrs.append({"key": "w3c.tracestate", "value": {"stringValue": envelope.w3c_tracestate}})
    return attrs


def _trace_row_tags(event: Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        tags = event.get("tags")
        return dict(tags) if isinstance(tags, dict) else {}
    tags = attribute_access.optional(event, "tags", None)
    return dict(tags) if isinstance(tags, dict) else {}


def _otlp_hex_id(value: str, *, length: int) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest[:length]


def _timestamp_to_unix_nano(value: Any) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return 0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    return 0
