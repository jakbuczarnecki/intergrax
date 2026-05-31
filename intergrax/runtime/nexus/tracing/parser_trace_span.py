# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Persist document parser traces as Nexus run trace events."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel, utc_now_iso


def append_parser_trace_event(
    trace_writer: RunTraceWriter,
    *,
    run_id: str,
    source: str,
    trace: dict[str, Any],
    session_id: str = "",
    user_id: str = "",
    tenant_id: str = "",
    seq: int = 0,
) -> None:
    """Write a structured parser pipeline trace span to the run trace store."""
    parser_id = trace.get("parser_id")
    event = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id=run_id,
        seq=seq,
        ts_utc=utc_now_iso(),
        level=TraceLevel.INFO,
        component=TraceComponent.RAG,
        step="document_parser",
        message=f"document_parser:{parser_id or 'unknown'}",
        payload=None,
        tags={
            "source": source,
            "integration_parser_trace": trace,
            "parser_id": parser_id,
            "session_id": session_id or None,
            "user_id": user_id or None,
            "tenant_id": tenant_id or None,
        },
    )
    trace_writer.append_event(event)


def maybe_append_parser_trace(
    trace_writer: Optional[RunTraceWriter],
    *,
    run_id: Optional[str],
    source: str,
    trace: dict[str, Any],
    session_id: str = "",
    user_id: str = "",
    tenant_id: str = "",
) -> None:
    if trace_writer is None or not run_id:
        return
    append_parser_trace_event(
        trace_writer,
        run_id=run_id,
        source=source,
        trace=trace,
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id,
    )
