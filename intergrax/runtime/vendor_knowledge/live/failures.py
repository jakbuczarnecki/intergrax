# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.vendor_knowledge.live.errors import LiveErrorCodeV1


class LiveCallFailureReasonV1(StrEnum):
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"


class LiveCallFailureV1(BaseModel):
    """Structural per-call execution failure without provider diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(..., min_length=1, max_length=128)
    reason: LiveCallFailureReasonV1


_AUTHORITY_UNAVAILABLE_CODES = frozenset({LiveErrorCodeV1.BINDING_UNAVAILABLE})

_PROVIDER_RESPONSE_INVALID_CODES = frozenset(
    {
        LiveErrorCodeV1.PROVIDER_CONTRACT_VIOLATION,
        LiveErrorCodeV1.RESULT_INVALID,
    }
)


def live_call_failure_reason_for_error_code(error_code: str) -> LiveCallFailureReasonV1:
    try:
        code = LiveErrorCodeV1(error_code)
    except ValueError:
        return LiveCallFailureReasonV1.PROVIDER_FAILED
    if code in _AUTHORITY_UNAVAILABLE_CODES:
        return LiveCallFailureReasonV1.AUTHORITY_UNAVAILABLE
    if code in _PROVIDER_RESPONSE_INVALID_CODES:
        return LiveCallFailureReasonV1.PROVIDER_RESPONSE_INVALID
    return LiveCallFailureReasonV1.PROVIDER_FAILED


def live_call_failure_for_error_code(
    *,
    call_id: str,
    error_code: str,
) -> LiveCallFailureV1:
    return LiveCallFailureV1(
        call_id=call_id,
        reason=live_call_failure_reason_for_error_code(error_code),
    )
