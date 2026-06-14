# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified run journal export — ref payloads and OTLP-style snapshots (OBS-BUS-6)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.unified_run_journal import (
    JOURNAL_SCHEMA_VERSION,
    build_unified_run_journal,
)
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun

JOURNAL_EXPORT_SCHEMA_VERSION = "journal_export.v1"


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class JournalExportSnapshot:
    """Full export snapshot for OTLP dual-write and operator tooling."""

    schema_version: str
    journal_schema_version: str
    run_id: str
    tenant_id: str
    event_count: int
    parser_trace_count: int
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "journal_schema_version": self.journal_schema_version,
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "event_count": self.event_count,
            "parser_trace_count": self.parser_trace_count,
            "events": self.events,
        }


def build_journal_ref(
    persisted: PersistedRun,
    *,
    runtime_store: RuntimeEventPersistence | None = None,
    limit: int = 2000,
) -> JournalRef | None:
    """Build a lightweight journal reference for terminal runtime events."""
    journal = build_unified_run_journal(persisted, runtime_store=runtime_store, limit=limit)
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
    runtime_store: RuntimeEventPersistence | None = None,
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
    runtime_store: RuntimeEventPersistence | None = None,
    limit: int = 2000,
) -> JournalExportSnapshot:
    """Serialize the unified journal for export sinks."""
    journal = build_unified_run_journal(persisted, runtime_store=runtime_store, limit=limit)
    return JournalExportSnapshot(
        schema_version=JOURNAL_EXPORT_SCHEMA_VERSION,
        journal_schema_version=JOURNAL_SCHEMA_VERSION,
        run_id=persisted.metadata.run_id,
        tenant_id=persisted.metadata.tenant_id,
        event_count=len(journal),
        parser_trace_count=count_parser_traces_in_trace_events(persisted.events),
        events=[serialize_runtime_event(event) for event in journal],
    )


def serialize_runtime_event(event: RuntimeEvent) -> Dict[str, Any]:
    return event.model_dump(mode="json")


def count_parser_traces_in_trace_events(events: Sequence[Any]) -> int:
    """Count persisted trace rows carrying ``integration_parser_trace`` tags."""
    count = 0
    for event in events:
        tags = _trace_row_tags(event)
        trace = tags.get("integration_parser_trace")
        if isinstance(trace, dict):
            count += 1
    return count


def render_journal_otlp_json(snapshot: JournalExportSnapshot | Mapping[str, Any]) -> Dict[str, Any]:
    """
    OTLP-inspired JSON trace snapshot for observability backends / debug export.

    Not a full OTLP protobuf encoder — stable JSON for log sinks and HTTP routes.
    """
    if isinstance(snapshot, JournalExportSnapshot):
        payload = snapshot.to_dict()
    else:
        payload = dict(snapshot)
    run_id = str(payload.get("run_id", ""))
    tenant_id = str(payload.get("tenant_id", ""))
    events = payload.get("events") or []
    spans: List[Dict[str, Any]] = []
    for row in events:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id", ""))
        event_type = str(row.get("event_type", "unknown"))
        spans.append(
            {
                "traceId": _otlp_hex_id(run_id, length=32),
                "spanId": _otlp_hex_id(event_id, length=16),
                "name": event_type,
                "kind": "SPAN_KIND_INTERNAL",
                "startTimeUnixNano": _timestamp_to_unix_nano(row.get("timestamp")),
                "attributes": _span_attributes(row, tenant_id=tenant_id),
            }
        )
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


def _span_attributes(row: Mapping[str, Any], *, tenant_id: str) -> List[Dict[str, Any]]:
    attrs: List[Dict[str, Any]] = [
        {"key": "intergrax.event_id", "value": {"stringValue": str(row.get("event_id", ""))}},
        {"key": "intergrax.tenant_id", "value": {"stringValue": tenant_id}},
        {"key": "intergrax.task_id", "value": {"stringValue": str(row.get("task_id", ""))}},
        {"key": "intergrax.phase", "value": {"stringValue": str(row.get("phase", ""))}},
    ]
    agent_id = row.get("agent_id")
    if agent_id:
        attrs.append({"key": "intergrax.agent_id", "value": {"stringValue": str(agent_id)}})
    parent = row.get("parent_event_id")
    if parent:
        attrs.append({"key": "intergrax.parent_event_id", "value": {"stringValue": str(parent)}})
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
