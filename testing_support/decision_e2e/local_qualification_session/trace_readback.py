# © Artur Czarnecki. All rights reserved.

"""Typed alignment diagnostic readback from persisted trace events."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.contracts import (
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    TraceReadbackStatus,
    TypedAlignmentEvent,
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


def _parse_alignment_event(payload: dict[str, object]) -> TypedAlignmentEvent | None:
    required_bool = (
        "alignment_mismatch_detected",
        "alignment_correctable",
        "alignment_correction_attempted",
        "alignment_correction_succeeded",
        "alignment_correction_exhausted",
        "revision_authoritative_context_present",
    )
    for key in required_bool:
        if key not in payload or not isinstance(payload[key], bool):
            return None
    direction = payload.get("alignment_direction")
    direction_value = direction if isinstance(direction, str) else None
    return TypedAlignmentEvent(
        alignment_mismatch_detected=bool(payload["alignment_mismatch_detected"]),
        alignment_direction=direction_value,
        alignment_correctable=bool(payload["alignment_correctable"]),
        alignment_correction_attempted=bool(payload["alignment_correction_attempted"]),
        alignment_correction_succeeded=bool(payload["alignment_correction_succeeded"]),
        alignment_correction_exhausted=bool(payload["alignment_correction_exhausted"]),
        revision_authoritative_context_present=bool(
            payload["revision_authoritative_context_present"]
        ),
    )


def read_typed_alignment_events(
    events: tuple[dict[str, object], ...],
    *,
    trace_available: bool = True,
) -> TypedAlignmentReadback:
    if not trace_available:
        return TypedAlignmentReadback(status=TraceReadbackStatus.NOT_AVAILABLE, events=())

    parsed: list[TypedAlignmentEvent] = []
    parse_failures = 0
    for event in events:
        if _event_schema_id(event) != COMPLETION_ALIGNMENT_TRACE_SCHEMA:
            continue
        alignment = _parse_alignment_event(_event_payload(event))
        if alignment is None:
            parse_failures += 1
            continue
        parsed.append(alignment)

    if parse_failures > 0:
        return TypedAlignmentReadback(status=TraceReadbackStatus.FAILED, events=())

    return TypedAlignmentReadback(status=TraceReadbackStatus.PASS, events=tuple(parsed))
