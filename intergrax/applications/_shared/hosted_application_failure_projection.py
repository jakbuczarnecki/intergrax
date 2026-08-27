# © Artur Czarnecki. All rights reserved.

"""Bounded HostedApplication failure → PlatformProblemSignal projection (HOST-DIAG-3)."""

from __future__ import annotations

from intergrax.contracts.event_severity import EventSeverity
from intergrax.hosting.contracts.events import (
    HostedApplicationEvent,
    HostedApplicationEventType,
)
from intergrax.hosting.eventing import HostingObservabilityAttributes
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE,
    PROBLEM_SEVERITY_CRITICAL,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SOURCE_LAYER_APPLICATION,
    PROBLEM_STATUS_DETECTED,
    PlatformProblemSignal,
)


def _bounded_payload_string(payload: dict[str, object], field_name: str) -> str | None:
    raw = payload.get(field_name)
    if raw is None:
        return None
    if type(raw) is not str:
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    return normalized


def hosted_application_failure_to_problem_signal(
    event: HostedApplicationEvent,
) -> PlatformProblemSignal | None:
    """
    Deterministic projection from canonical APPLICATION_FAILED hosting truth.

    Returns ``None`` for non-failure events or events without bounded failure facts.
    """
    if event.event_type is not HostedApplicationEventType.APPLICATION_FAILED:
        return None

    payload = event.payload
    if not payload:
        return None

    phase = _bounded_payload_string(payload, "phase")
    reason_code = _bounded_payload_string(payload, "reason_code")
    source_kind = _bounded_payload_string(payload, "source_kind")
    source_id = _bounded_payload_string(payload, "source_id")
    exception_type = _bounded_payload_string(payload, "exception_type")
    if phase is None or reason_code is None:
        return None

    source_component = phase
    severity = (
        PROBLEM_SEVERITY_CRITICAL
        if event.severity is EventSeverity.CRITICAL
        else PROBLEM_SEVERITY_ERROR
    )
    application_attributes = HostingObservabilityAttributes(
        application_id=event.application_id,
        instance_id=event.instance_id,
        lifecycle_state=event.lifecycle_state.value,
        severity=event.severity.value,
        occurred_at=event.occurred_at.isoformat(),
        causation_id=event.causation_id,
    )

    return PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_APPLICATION_FAILURE,
        severity=severity,
        source_layer=PROBLEM_SOURCE_LAYER_APPLICATION,
        source_component=source_component,
        status=PROBLEM_STATUS_DETECTED,
        error_code=reason_code,
        exception_type=exception_type,
        safe_message="hosted application failure",
        event_id=event.event_id,
        correlation_id=event.correlation_id or event.event_id,
        occurred_at=event.occurred_at,
        application_attributes=application_attributes,
    )


__all__ = [
    "hosted_application_failure_to_problem_signal",
]
