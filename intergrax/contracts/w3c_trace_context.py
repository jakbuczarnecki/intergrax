# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W3C Trace Context validation for canonical RuntimeEvent (OBS-EVOL-9.11)."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>00)-(?P<trace_id>[0-9a-f]{32})-(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
_TRACESTATE_RE = re.compile(r"^[\x20-\x7e]*$")
_MAX_TRACESTATE_LEN = 512


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


def _validate_trace_id(trace_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", trace_id):
        raise W3CTraceContextError(
            f"trace_id must be 32 lowercase hex chars, got {trace_id!r}"
        )
    if int(trace_id, 16) == 0:
        raise W3CTraceContextError("trace_id must be non-zero")


def _validate_span_id(span_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{16}", span_id):
        raise W3CTraceContextError(
            f"parent_id must be 16 lowercase hex chars, got {span_id!r}"
        )
    if int(span_id, 16) == 0:
        raise W3CTraceContextError("parent_id must be non-zero")


__all__ = [
    "ParsedTraceParent",
    "W3CTraceContextError",
    "child_traceparent",
    "format_traceparent",
    "generate_span_id",
    "is_valid_traceparent",
    "is_valid_tracestate",
    "parse_traceparent",
]
