# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution-attempt retry contracts (NPSC-5E/R1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.resilience_policy import FailureClass


class ExecutionFailureKind(StrEnum):
    """Typed execution-attempt failure projection for retry eligibility."""

    RETRYABLE_TRANSIENT = "retryable_transient"
    RETRYABLE_TIMEOUT = "retryable_timeout"
    NON_RETRYABLE_PERMANENT = "non_retryable_permanent"
    GOVERNANCE_DENIED = "governance_denied"
    AUTHORITY_DENIED = "authority_denied"
    TRUST_DENIED = "trust_denied"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_DENY = "terminal_deny"
    CONTRACT_ERROR = "contract_error"
    UNKNOWN_UNSAFE = "unknown_unsafe"
    UNKNOWN = "unknown"


class ExecutionRetryAction(StrEnum):
    """Canonical recovery decision for execution-attempt retry."""

    FAIL = "fail"
    RETRY = "retry"
    CANCEL = "cancel"


class ExecutionFailureClassification(BaseModel):
    """Immutable typed failure classification for execution retry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ExecutionFailureKind
    reason: str = Field(default="", max_length=512)
    failure_class: FailureClass | None = None
    has_unknown_side_effect: bool = False


class ExecutionRetryEligibilityRequest(BaseModel):
    """Inputs required to decide whether a new execution attempt may start."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: ExecutionFailureClassification
    attempt_number: int = Field(ge=1, le=64)
    max_attempts: int = Field(ge=1, le=64)
    cancelled: bool = False
    terminal_outcome: ExecutionTerminalOutcome | None = None
    global_deadline_monotonic: float | None = None
    now_monotonic: float | None = None
    proposed_backoff_seconds: float = Field(default=0.0, ge=0.0)
    side_effect_idempotency_guaranteed: bool = False


class ExecutionRetryEligibilityResult(BaseModel):
    """Typed retry eligibility outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ExecutionRetryAction
    reason: str = Field(default="", max_length=512)
    backoff_delay_seconds: float = Field(default=0.0, ge=0.0)


class BackoffKind(StrEnum):
    """Replaceable backoff policy semantics."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    JITTERED = "jittered"
    NONE = "none"


class BackoffPolicyConfig(BaseModel):
    """Bounded backoff configuration for execution-attempt retry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BackoffKind = BackoffKind.EXPONENTIAL
    base_delay_seconds: float = Field(default=0.25, ge=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_seconds: float = Field(default=30.0, gt=0.0)
    jitter_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
