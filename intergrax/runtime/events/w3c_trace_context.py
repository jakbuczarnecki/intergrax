# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W3C Trace Context helpers for RuntimeEvent correlation (OBS-EVOL-9.11)."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>00)-(?P<trace_id>[0-9a-f]{32})-(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_TRACESTATE_RE = re.compile(r"^[\x20-\x7e]*$")
_MAX_TRACESTATE_LEN = 512

_W3C_TRACE_ID_KEY = "w3c_trace_id"
_W3C_TRACESTATE_KEY = "w3c_tracestate"
_INBOUND_TRACEPARENT_KEYS = ("traceparent", "w3c_traceparent")


class W3CTraceContextError(ValueError):
    """Raised when a W3C trace context value is invalid."""


@dataclass(frozen=True, slots=True)
class ParsedTraceParent:
    version: str
    trace_id: str
    parent_id: str
    flags: str

    @property
    def sampled(self) -> bool:
        return (int(self.flags, 16) & 0x01) == 1


@dataclass(frozen=True, slots=True)
class RunTraceContext:
    """Stable trace id for a Nexus run plus optional vendor tracestate."""

    trace_id: str
    tracestate: str | None = None


def generate_trace_id() -> str:
    return secrets.token_hex(16)


def generate_span_id() -> str:
    return secrets.token_hex(8)


def format_traceparent(
    *,
    trace_id: str,
    parent_id: str,
    sampled: bool = True,
    version: str = "00",
) -> str:
    _validate_trace_id(trace_id)
    _validate_span_id(parent_id)
    flags = "01" if sampled else "00"
    return f"{version}-{trace_id}-{parent_id}-{flags}"


def parse_traceparent(value: str) -> ParsedTraceParent:
    match = _TRACEPARENT_RE.match(value.strip())
    if match is None:
        raise W3CTraceContextError(f"invalid traceparent: {value!r}")
    trace_id = match.group("trace_id")
    if int(trace_id, 16) == 0:
        raise W3CTraceContextError("traceparent trace_id must be non-zero")
    parent_id = match.group("parent_id")
    if int(parent_id, 16) == 0:
        raise W3CTraceContextError("traceparent parent_id must be non-zero")
    return ParsedTraceParent(
        version=match.group("version"),
        trace_id=trace_id,
        parent_id=parent_id,
        flags=match.group("flags"),
    )


def is_valid_traceparent(value: str) -> bool:
    try:
        parse_traceparent(value)
    except W3CTraceContextError:
        return False
    return True


def is_valid_tracestate(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if len(text) > _MAX_TRACESTATE_LEN:
        return False
    return _TRACESTATE_RE.match(text) is not None


def child_traceparent(parent: str, *, sampled: bool | None = None) -> str:
    parsed = parse_traceparent(parent)
    return format_traceparent(
        trace_id=parsed.trace_id,
        parent_id=generate_span_id(),
        sampled=parsed.sampled if sampled is None else sampled,
        version=parsed.version,
    )


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


def trace_context_from_metadata(metadata: Mapping[str, Any]) -> tuple[str | None, str | None]:
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
    metadata = getattr(task, "metadata", None)
    if not isinstance(metadata, dict):
        return RunTraceContext(trace_id=generate_trace_id())

    existing = metadata.get(_W3C_TRACE_ID_KEY)
    if isinstance(existing, str) and len(existing) == 32 and re.fullmatch(r"[0-9a-f]{32}", existing):
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
    if getattr(event, "traceparent", None):
        return event
    run_ctx = ensure_run_trace_context(task)
    return event.model_copy(
        update={
            "traceparent": traceparent_for_run_span(run_ctx),
            "tracestate": run_ctx.tracestate,
        }
    )


def _validate_trace_id(trace_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        raise W3CTraceContextError(f"trace_id must be 32 lowercase hex chars, got {trace_id!r}")
    if int(trace_id, 16) == 0:
        raise W3CTraceContextError("trace_id must be non-zero")


def _validate_span_id(span_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{16}", span_id):
        raise W3CTraceContextError(f"parent_id must be 16 lowercase hex chars, got {span_id!r}")
    if int(span_id, 16) == 0:
        raise W3CTraceContextError("parent_id must be non-zero")
