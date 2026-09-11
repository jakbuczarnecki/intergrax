# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider rate limit contract (W2-C).

Answers only whether a physical provider attempt may start now (throughput over time).
Does not implement retry, circuit breaking, or admission.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.retry_budget import RetryBudgetIdentity, RetryBudgetKind


def _validate_provider_rate_limit_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("provider rate limit value must be str")
    if value == "":
        raise ValueError("provider rate limit value must not be empty")
    if value != value.strip():
        raise ValueError("provider rate limit value must equal its stripped form")
    return value


class ProviderRateLimitError(RuntimeError):
    """Base error for provider rate limiting."""


class ProviderRateLimitExceededError(ProviderRateLimitError):
    """No capacity right now under REJECT overload policy."""


class ProviderRateLimitWaitTimeoutError(ProviderRateLimitError):
    """Capacity did not become available before WAIT_WITH_TIMEOUT elapsed."""


class ProviderRateLimitPolicyMissingError(ProviderRateLimitError):
    """Rate limit port is active but no policy is configured for the identity."""


class ProviderRateLimitOverloadMode(StrEnum):
    REJECT = "REJECT"
    WAIT_WITH_TIMEOUT = "WAIT_WITH_TIMEOUT"


class ProviderRateLimitIdentity(BaseModel):
    """Provider-scoped rate limit key (LLM_PROVIDER + slug)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RetryBudgetKind
    value: str

    @model_validator(mode="after")
    def _validate_value(self) -> ProviderRateLimitIdentity:
        object.__setattr__(self, "value", _validate_provider_rate_limit_value(self.value))
        return self

    @classmethod
    def from_retry_budget_identity(
        cls,
        identity: RetryBudgetIdentity,
    ) -> ProviderRateLimitIdentity:
        return cls(kind=identity.kind, value=identity.value)


class ProviderRateLimitPolicy(BaseModel):
    """Explicit provider rate limit policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls_per_minute: int = Field(ge=1)
    overload_mode: ProviderRateLimitOverloadMode
    wait_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_wait_timeout(self) -> ProviderRateLimitPolicy:
        if self.overload_mode is ProviderRateLimitOverloadMode.WAIT_WITH_TIMEOUT:
            if self.wait_timeout_seconds is None:
                raise ValueError(
                    "wait_timeout_seconds must be set when overload_mode is WAIT_WITH_TIMEOUT"
                )
        elif self.wait_timeout_seconds is not None:
            raise ValueError(
                "wait_timeout_seconds applies only when overload_mode is WAIT_WITH_TIMEOUT"
            )
        return self


@runtime_checkable
class ProviderRateLimitPort(Protocol):
    """Pluginable provider rate limiter."""

    def acquire_for_physical_attempt(
        self,
        identity: ProviderRateLimitIdentity,
        policy: ProviderRateLimitPolicy,
    ) -> None:
        """Consume one attempt slot now or fail fast / time out per policy."""
        ...
