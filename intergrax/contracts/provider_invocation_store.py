# © Artur Czarnecki. All rights reserved.

"""Durable provider invocation intent and outcome port (GR-7-A3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)


class ProviderInvocationIntentWriteResult(StrEnum):
    """Atomic first-writer admission fact for dispatch eligibility (ADR-03 §2.3)."""

    INSERTED = "INSERTED"
    EQUIVALENT_NOOP = "EQUIVALENT_NOOP"


class ProviderInvocationStoreError(RuntimeError):
    """Base error for provider invocation durability."""


class ProviderInvocationConflictError(ProviderInvocationStoreError):
    """Raised when the same invocation_id is bound to a different payload."""


class ProviderInvocationOutcomeConflictError(ProviderInvocationStoreError):
    """Raised when a different final outcome exists for the same invocation."""


class ProviderInvocationPersistenceError(ProviderInvocationStoreError):
    """Raised when durable storage is unavailable or fails."""


def provider_invocations_equivalent(
    left: ProviderInvocation,
    right: ProviderInvocation,
) -> bool:
    """Idempotency: same invocation identity and canonical payload."""
    return left == right


def provider_invocation_outcomes_equivalent(
    left: ProviderInvocationOutcome,
    right: ProviderInvocationOutcome,
) -> bool:
    """Idempotency: same outcome record."""
    return left == right


class ProviderInvocationStore(ABC):
    """Platform-owned, provider-neutral durable invocation lifecycle port."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether records survive process restart."""

    @abstractmethod
    def put_invocation(self, invocation: ProviderInvocation) -> None:
        """
        Persist intent before provider dispatch.

        Repeated put of an equivalent invocation is a no-op. A conflicting
        payload for the same ``invocation_id`` fails closed.
        """

    @abstractmethod
    def insert_invocation_if_absent(
        self,
        invocation: ProviderInvocation,
    ) -> ProviderInvocationIntentWriteResult:
        """
        Atomically insert intent or return ``EQUIVALENT_NOOP`` when equivalent.

        Conflicting payload for the same ``invocation_id`` raises
        ``ProviderInvocationConflictError``. Infrastructure failures raise
        ``ProviderInvocationPersistenceError``.
        """

    @abstractmethod
    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        """Load invocation intent or ``None``."""

    @abstractmethod
    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        """
        Persist observed provider outcome after dispatch.

        Repeated put of an equivalent outcome is a no-op. A conflicting final
        outcome for the same ``invocation_id`` fails closed.
        """

    @abstractmethod
    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        """Load provider outcome or ``None``."""


__all__ = [
    "ProviderInvocationConflictError",
    "ProviderInvocationIntentWriteResult",
    "ProviderInvocationOutcomeConflictError",
    "ProviderInvocationPersistenceError",
    "ProviderInvocationStore",
    "ProviderInvocationStoreError",
    "provider_invocation_outcomes_equivalent",
    "provider_invocations_equivalent",
]
