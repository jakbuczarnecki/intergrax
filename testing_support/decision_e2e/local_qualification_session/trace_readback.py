# © Artur Czarnecki. All rights reserved.

"""Typed alignment diagnostic readback from persisted trace events."""

from __future__ import annotations

from intergrax.runtime.diagnostics.completion_alignment_diag import (
    CompletionAlignmentDiagV1,
    decode_completion_alignment_diag_v1,
)

from testing_support.decision_e2e.local_qualification_session.contracts import (
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    TraceReadbackStatus,
    TypedAlignmentReadback,
)


def _event_schema_id(event: dict[str, object]) -> str | None:
    schema = event.get("payload_schema_id")
    if isinstance(schema, str):
        return schema
    return None


def _event_payload(event: dict[str, object]) -> dict[str, object]:
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def read_typed_alignment_events(
    events: tuple[dict[str, object], ...],
    *,
    trace_available: bool = True,
) -> TypedAlignmentReadback:
    if not trace_available:
        return TypedAlignmentReadback(status=TraceReadbackStatus.NOT_AVAILABLE, events=())

    parsed: list[CompletionAlignmentDiagV1] = []
    parse_failures = 0
    for event in events:
        if _event_schema_id(event) != COMPLETION_ALIGNMENT_TRACE_SCHEMA:
            continue
        decoded = decode_completion_alignment_diag_v1(_event_payload(event))
        if decoded is None:
            parse_failures += 1
            continue
        parsed.append(decoded)

    if parse_failures > 0:
        return TypedAlignmentReadback(status=TraceReadbackStatus.FAILED, events=())

    return TypedAlignmentReadback(status=TraceReadbackStatus.PASS, events=tuple(parsed))
