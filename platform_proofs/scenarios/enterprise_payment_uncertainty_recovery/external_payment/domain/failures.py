"""Communication and infrastructure failures at the external integration boundary."""

from __future__ import annotations

from enum import StrEnum


class CommunicationFailureKind(StrEnum):
    """Failures on the wire — not equivalent to payment business failure."""

    NETWORK_INTERRUPTION = "NETWORK_INTERRUPTION"
    EXTERNAL_TIMEOUT = "EXTERNAL_TIMEOUT"
    UNAVAILABLE_RESPONSE = "UNAVAILABLE_RESPONSE"
    RESPONSE_LOST = "RESPONSE_LOST"
    PROCESSING_DELAY = "PROCESSING_DELAY"


_FAILURE_KIND_BY_UNCERTAINTY_CAUSE: dict[str, CommunicationFailureKind] = {
    "response_lost_after_request_accepted": CommunicationFailureKind.RESPONSE_LOST,
    "communication_interruption_prevents_authoritative_query": (
        CommunicationFailureKind.NETWORK_INTERRUPTION
    ),
    "external_timeout_before_response": CommunicationFailureKind.EXTERNAL_TIMEOUT,
    "provider_unavailable": CommunicationFailureKind.UNAVAILABLE_RESPONSE,
    "processing_delay_exceeded_client_window": CommunicationFailureKind.PROCESSING_DELAY,
}


def resolve_communication_failure_kind(uncertainty_cause: str) -> CommunicationFailureKind:
    mapped = _FAILURE_KIND_BY_UNCERTAINTY_CAUSE.get(uncertainty_cause)
    if mapped is not None:
        return mapped
    return CommunicationFailureKind.UNAVAILABLE_RESPONSE
