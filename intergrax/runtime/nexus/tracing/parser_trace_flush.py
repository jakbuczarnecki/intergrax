# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Flush parser trace tags from finalized Nexus runs into structured logging."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.contracts.persisted_run_trace import PersistedTraceEvent, decode_persisted_trace_event
from intergrax.contracts.structured_json_value import StructuredJsonObject
from intergrax.contracts.tracing import TraceEvent, TraceObject
from intergrax.rag.document_loaders.observability.parser_trace_contract import parser_trace_from_tags
from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace
from intergrax.runtime.nexus.tracing.persistence_models import SerializedTraceEvent

ParserTraceFlushEvent = (
    TraceEvent | SerializedTraceEvent | PersistedTraceEvent | StructuredJsonObject
)


def export_parser_traces_from_events(events: Iterable[ParserTraceFlushEvent]) -> None:
    """Scan trace events for ``integration_parser_trace`` tags and emit structured logs."""
    for event in events:
        tags = _event_tags(event)
        trace = parser_trace_from_tags(tags)
        if trace is None:
            continue
        source_raw = tags.get("source")
        source = source_raw if isinstance(source_raw, str) and source_raw else "nexus_run_trace"
        export_parser_trace(source=source, trace=trace)


def _event_tags(event: ParserTraceFlushEvent) -> TraceObject:
    if isinstance(event, TraceEvent):
        return dict(event.tags)
    if isinstance(event, SerializedTraceEvent):
        return dict(event.tags)
    if isinstance(event, PersistedTraceEvent):
        return dict(event.tags) if event.tags is not None else {}
    return dict(decode_persisted_trace_event(event).tags or {})
