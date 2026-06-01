# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Export parser trace spans from finalized Nexus runs to observability vendors."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class TraceEventWithTags(Protocol):
    """Trace event carrying a tags mapping (Nexus trace persistence)."""

    tags: Mapping[str, Any]


from intergrax.rag.document_loaders.observability.parser_trace_exporter import export_parser_trace


def export_parser_traces_from_events(events: Iterable[Any]) -> None:
    """Scan trace events for ``integration_parser_trace`` tags and export to Langfuse/Sentry."""
    for event in events:
        tags = _event_tags(event)
        trace = tags.get("integration_parser_trace")
        if not isinstance(trace, dict):
            continue
        source = str(tags.get("source") or "nexus_run_trace")
        export_parser_trace(source=source, trace=trace)


def _event_tags(event: Any) -> dict[str, Any]:
    if isinstance(event, TraceEventWithTags):
        return dict(event.tags)
    if isinstance(event, dict):
        tags = event.get("tags")
        return dict(tags) if isinstance(tags, dict) else {}
    return {}
