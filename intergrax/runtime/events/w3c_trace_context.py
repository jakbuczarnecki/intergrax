# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W3C Trace Context helpers for RuntimeEvent correlation (OBS-EVOL-9.11)."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from intergrax.contracts.w3c_trace_context import (
    ParsedTraceParent,
    W3CTraceContextError,
    child_traceparent,
    format_traceparent,
    generate_span_id,
    is_valid_traceparent,
    is_valid_tracestate,
    parse_traceparent,
)
from intergrax.utils import attribute_access

_W3C_TRACE_ID_KEY = "w3c_trace_id"
_W3C_TRACESTATE_KEY = "w3c_tracestate"
_INBOUND_TRACEPARENT_KEYS = ("traceparent", "w3c_traceparent")


@dataclass(frozen=True, slots=True)
class RunTraceContext:
    """Stable trace id for a Nexus run plus optional vendor tracestate."""

    trace_id: str
    tracestate: str | None = None


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def traceparent_for_run_span(
    run_ctx: RunTraceContext,
    *,
    span_id: str | None = None,
    sampled: bool = True,
) -> str:
    return format_traceparent(
        trace_id=run_ctx.trace_id,
        parent_id=span_id or generate_span_id(),
        sampled=sampled,
    )


def trace_context_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Read inbound W3C headers stored on task metadata."""
    traceparent: str | None = None
    for key in _INBOUND_TRACEPARENT_KEYS:
        raw = metadata.get(key)
        if isinstance(raw, str) and is_valid_traceparent(raw):
            traceparent = raw.strip()
            break
    tracestate_raw = metadata.get("tracestate") or metadata.get("w3c_tracestate")
    tracestate = (
        tracestate_raw.strip()
        if isinstance(tracestate_raw, str) and is_valid_tracestate(tracestate_raw)
        else None
    )
    return traceparent, tracestate


def ensure_run_trace_context(task: Any) -> RunTraceContext:
    """Resolve or allocate a stable W3C trace id for a Nexus task run."""
    metadata = attribute_access.optional(task, "metadata", None)
    if not isinstance(metadata, dict):
        return RunTraceContext(trace_id=generate_trace_id())

    existing = metadata.get(_W3C_TRACE_ID_KEY)
    if (
        isinstance(existing, str)
        and len(existing) == 32
        and re.fullmatch(r"[0-9a-f]{32}", existing)
    ):
        tracestate = metadata.get(_W3C_TRACESTATE_KEY)
        return RunTraceContext(
            trace_id=existing,
            tracestate=tracestate if isinstance(tracestate, str) else None,
        )

    inbound_tp, inbound_ts = trace_context_from_metadata(metadata)
    if inbound_tp is not None:
        parsed = parse_traceparent(inbound_tp)
        trace_id = parsed.trace_id
        tracestate = inbound_ts
    else:
        trace_id = generate_trace_id()
        tracestate = inbound_ts

    metadata[_W3C_TRACE_ID_KEY] = trace_id
    if tracestate is not None:
        metadata[_W3C_TRACESTATE_KEY] = tracestate
    return RunTraceContext(trace_id=trace_id, tracestate=tracestate)


def inject_w3c_trace_on_event(event: Any, task: Any) -> Any:
    """Attach a per-event traceparent when the event does not already carry W3C context."""
    if attribute_access.optional(event, "traceparent", None):
        return event
    run_ctx = ensure_run_trace_context(task)
    return event.model_copy(
        update={
            "traceparent": traceparent_for_run_span(run_ctx),
            "tracestate": run_ctx.tracestate,
        }
    )


__all__ = [
    "ParsedTraceParent",
    "RunTraceContext",
    "W3CTraceContextError",
    "child_traceparent",
    "ensure_run_trace_context",
    "format_traceparent",
    "generate_span_id",
    "generate_trace_id",
    "inject_w3c_trace_on_event",
    "is_valid_traceparent",
    "is_valid_tracestate",
    "parse_traceparent",
    "trace_context_from_metadata",
    "traceparent_for_run_span",
]
