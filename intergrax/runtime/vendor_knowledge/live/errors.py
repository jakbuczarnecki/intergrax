from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LiveErrorCodeV1(StrEnum):
    BINDING_UNAVAILABLE = "live_binding_unavailable"
    CAPABILITY_UNAVAILABLE = "live_capability_unavailable"
    REQUEST_INVALID = "live_request_invalid"
    RESOURCE_SCOPE_INVALID = "live_resource_scope_invalid"
    EXECUTION_TIMEOUT = "live_execution_timeout"
    EXECUTION_FAILED = "live_execution_failed"
    RESULT_INVALID = "live_result_invalid"
    RESULT_TOO_LARGE = "live_result_too_large"
    PROVIDER_UNAUTHORIZED = "live_provider_unauthorized"
    PROVIDER_FORBIDDEN = "live_provider_forbidden"
    PROVIDER_NOT_FOUND = "live_provider_not_found"
    PROVIDER_THROTTLED = "live_provider_throttled"
    PROVIDER_TEMPORARILY_UNAVAILABLE = "live_provider_temporarily_unavailable"
    PROVIDER_CONTRACT_VIOLATION = "live_provider_contract_violation"


_RETRYABLE_CODES = frozenset(
    {
        LiveErrorCodeV1.EXECUTION_TIMEOUT,
        LiveErrorCodeV1.EXECUTION_FAILED,
        LiveErrorCodeV1.PROVIDER_THROTTLED,
        LiveErrorCodeV1.PROVIDER_TEMPORARILY_UNAVAILABLE,
    }
)


class LiveErrorEnvelopeV1(BaseModel):
    """Public code plus bounded retry metadata; no provider diagnostic payload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    code: LiveErrorCodeV1
    retryable: bool
    diagnostic_ref: str | None = Field(default=None, max_length=128)


def retryable_for_live_error(code: LiveErrorCodeV1) -> bool:
    return code in _RETRYABLE_CODES


def normalize_live_error(code: LiveErrorCodeV1) -> LiveErrorEnvelopeV1:
    return LiveErrorEnvelopeV1(
        code=code,
        retryable=retryable_for_live_error(code),
    )
