# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative decision event append contracts (W3-C / ADR Option C).

Compare-and-append only — no storage backend in this module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.contracts.decision_identity import DecisionId


class StaleDecisionEventAppendError(RuntimeError):
    """Authoritative decision event history conflict on compare-and-append."""


class DuplicateDecisionEventError(RuntimeError):
    """Raised when the same ``event_id`` is reused with a different typed payload."""


@runtime_checkable
class DecisionEventPayload(Protocol):
    """Immutable typed payload or reference carried by one decision event."""


@runtime_checkable
class DecisionEvent(Protocol):
    """One immutable decision lifecycle event in the authoritative stream."""

    @property
    def event_id(self) -> str:
        """Stable identifier for idempotent replay or duplicate rejection."""
        ...

    @property
    def decision_id(self) -> DecisionId:
        """Decision identity this event belongs to."""
        ...

    @property
    def event_sequence(self) -> int:
        """Monotonic sequence assigned by the store on successful append (0 before persist)."""
        ...

    @property
    def event_type(self) -> str:
        """Stable event kind for audit and replay."""
        ...

    @property
    def occurred_at_utc(self) -> str:
        """ISO-8601 UTC timestamp for audit."""
        ...

    @property
    def payload(self) -> DecisionEventPayload:
        """Typed event body — not a generic mapping."""
        ...


@runtime_checkable
class DecisionEventAppendPort(Protocol):
    """Execution-hosted durable decision event stream (compare-and-append)."""

    def append(
        self,
        *,
        key: DecisionFinalizationKey,
        event: DecisionEvent,
        expected_last_sequence: int,
    ) -> DecisionEvent:
        """Append one event when ``current_last_sequence == expected_last_sequence``.

        On success the returned event carries ``event_sequence == expected_last_sequence + 1``.
        On conflict raises :class:`StaleDecisionEventAppendError`.
        Retry with the same ``event_id`` and payload must be idempotent replay or raise
        :class:`DuplicateDecisionEventError` when the payload differs — without a global
        idempotency manager.
        """
        ...
