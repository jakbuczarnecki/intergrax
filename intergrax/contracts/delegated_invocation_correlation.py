# © Artur Czarnecki. All rights reserved.

"""Durable delegated invocation correlation (P2.1-S2C).

Maps canonical child ``ExecutionId`` to platform-issued
``DelegatedExecutionInvocationBinding`` without conflating provider-native
invocation identity with execution identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.contracts.delegated_execution_invocation_binding import (
    DelegatedExecutionInvocationBinding,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id

SCHEMA_DELEGATED_INVOCATION_CORRELATION_V1: Final = "delegated_invocation_correlation.v1"


class DelegatedInvocationCorrelationError(RuntimeError):
    """Base error for delegated invocation correlation persistence."""


class DelegatedInvocationCorrelationConflictError(DelegatedInvocationCorrelationError):
    """Raised when a different provider invocation is bound to the same execution."""


class DelegatedInvocationCorrelationIntegrityError(DelegatedInvocationCorrelationError):
    """Raised when stored correlation evidence fails validation."""


class DelegatedInvocationCorrelationPersistenceError(DelegatedInvocationCorrelationError):
    """Raised when durable correlation storage is unavailable or fails."""


class DelegatedInvocationCorrelationCompositionError(DelegatedInvocationCorrelationError):
    """Raised when correlation durability policy and store wiring are inconsistent."""


class DelegatedInvocationCorrelationDurabilityMode(StrEnum):
    """Explicit delegated invocation correlation durability posture at composition."""

    DISABLED = "disabled"
    REQUIRED = "required"
    NON_DURABLE_TEST = "non_durable_test"


class DelegatedInvocationCorrelationDurabilityPolicy(BaseModel):
    """Typed composition policy — no implicit constructor-arg durability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DelegatedInvocationCorrelationDurabilityMode = (
        DelegatedInvocationCorrelationDurabilityMode.DISABLED
    )


class DelegatedInvocationCorrelationRecord(BaseModel):
    """Immutable durable evidence: child execution ↔ platform-issued binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delegated_invocation_correlation.v1"] = (
        SCHEMA_DELEGATED_INVOCATION_CORRELATION_V1
    )
    binding: DelegatedExecutionInvocationBinding
    persisted_at: datetime

    @field_validator("persisted_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("persisted_at must be timezone-aware")
        return value


def correlation_records_equivalent(
    left: DelegatedInvocationCorrelationRecord,
    right: DelegatedInvocationCorrelationRecord,
) -> bool:
    """Idempotency: same execution, binding, and digest proof."""
    return left.binding == right.binding


class DelegatedInvocationCorrelationStore(ABC):
    """Platform-owned, provider-neutral durable correlation port."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether correlation survives process restart."""

    @abstractmethod
    def persist(self, record: DelegatedInvocationCorrelationRecord) -> None:
        """
        Append-once correlation for ``record.binding.execution_id``.

        Repeated persist of an equivalent record is a no-op. A different binding
        for the same execution_id fails closed.
        """

    @abstractmethod
    def get_by_execution_id(
        self,
        execution_id: ExecutionId,
    ) -> DelegatedInvocationCorrelationRecord | None:
        """Load correlation for a canonical child execution or ``None``."""

    def _validate_execution_id(self, execution_id: ExecutionId) -> ExecutionId:
        return validate_execution_id(execution_id)


__all__ = [
    "DelegatedInvocationCorrelationCompositionError",
    "DelegatedInvocationCorrelationConflictError",
    "DelegatedInvocationCorrelationDurabilityMode",
    "DelegatedInvocationCorrelationDurabilityPolicy",
    "DelegatedInvocationCorrelationError",
    "DelegatedInvocationCorrelationIntegrityError",
    "DelegatedInvocationCorrelationPersistenceError",
    "DelegatedInvocationCorrelationRecord",
    "DelegatedInvocationCorrelationStore",
    "SCHEMA_DELEGATED_INVOCATION_CORRELATION_V1",
    "correlation_records_equivalent",
]
