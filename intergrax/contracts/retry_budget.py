# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider retry budget contract (W2-C).

Answers only whether another physical provider attempt may begin for a logical call.
Does not implement retry policy, backoff, circuit breaking, admission, or rate limits.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryBudgetError(RuntimeError):
    """Base error for retry budget decisions."""


class RetryBudgetExhaustedError(RetryBudgetError):
    """No remaining attempts under the configured retry budget."""


class RetryBudgetPolicyMissingError(RetryBudgetError):
    """Retry budget port is active but no policy is configured for the identity."""


class RetryBudgetKind(StrEnum):
    """Typed retry budget domain (not tenant, not model)."""

    LLM_PROVIDER = "LLM_PROVIDER"


def _validate_retry_budget_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("retry budget value must be str")
    if value == "":
        raise ValueError("retry budget value must not be empty")
    if value != value.strip():
        raise ValueError("retry budget value must equal its stripped form")
    return value


class RetryBudgetIdentity(BaseModel):
    """Immutable provider-scoped retry budget key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: RetryBudgetKind
    value: str

    @model_validator(mode="after")
    def _validate_value(self) -> RetryBudgetIdentity:
        object.__setattr__(self, "value", _validate_retry_budget_value(self.value))
        return self


class RetryBudgetPolicy(BaseModel):
    """Explicit retry budget policy (no platform default)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts_per_logical_call: int = Field(ge=1)
    max_aggregate_physical_attempts: int | None = Field(default=None, ge=1)


@runtime_checkable
class RetryBudgetLogicalCall(Protocol):
    """Per logical provider call attempt ledger."""

    def begin_physical_attempt(self) -> None:
        """Reserve one physical attempt or raise ``RetryBudgetExhaustedError``."""
        ...

    def complete_physical_attempt(self) -> None:
        """Release aggregate reservations after one physical attempt finishes."""
        ...


@runtime_checkable
class RetryBudgetPort(Protocol):
    """Pluginable retry budget (process-local, distributed, etc.)."""

    def open_logical_call(
        self,
        identity: RetryBudgetIdentity,
        policy: RetryBudgetPolicy,
    ) -> RetryBudgetLogicalCall:
        """Start one logical call scope under the given explicit policy."""
        ...
