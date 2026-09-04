# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-hosted Decision durable finalization port (DS-REC-01).

Atomic compare-and-commit for terminal authoritative outcomes. Pure guard
semantics remain in ``intergrax.contracts.decision_finalization``; backends
wrap one transactional boundary around ``guard_decision_finalization``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.decision_finalization import (
    DecisionFinalizationConflictError,
    DecisionFinalizationKey,
    DecisionFinalizeDisposition,
    DecisionFinalizeGuardState,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord

T = TypeVar("T")


class DecisionDurableFinalizationDisposition(str, Enum):
    """Outcome of one atomic durable finalization attempt."""

    COMMITTED = "committed"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class DecisionDurableFinalizationResult(Generic[T]):
    """Typed result of one durable finalization commit."""

    disposition: DecisionDurableFinalizationDisposition
    guard_state: DecisionFinalizeGuardState[T]


class DecisionFinalizationPersistence(Protocol[T]):
    """Execution-facing atomic finalization durability port."""

    def load_guard_state(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionFinalizeGuardState[T] | None:
        """Return persisted guard state or ``None`` when no outcome exists."""
        ...

    def commit_authoritative_outcome(
        self,
        *,
        key: DecisionFinalizationKey,
        requested_outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
    ) -> DecisionDurableFinalizationResult[T]:
        """Atomically commit or reconcile one terminal authoritative outcome."""
        ...


def load_decision_finalization_guard_state(
    persistence: DecisionFinalizationPersistence[T],
    *,
    key: DecisionFinalizationKey,
) -> DecisionFinalizeGuardState[T]:
    """Load persisted guard state or return an unfinalized guard for one key."""
    loaded = persistence.load_guard_state(key=key)
    if loaded is None:
        return initial_decision_finalize_guard(key)
    if loaded.key != key:
        raise ValueError("persisted finalization guard key does not match requested key")
    return loaded


def commit_durable_authoritative_outcome(
    persistence: DecisionFinalizationPersistence[T],
    *,
    key: DecisionFinalizationKey,
    requested_outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
) -> DecisionDurableFinalizationResult[T]:
    """Commit one authoritative outcome through the durable finalization port."""
    return persistence.commit_authoritative_outcome(
        key=key,
        requested_outcome=requested_outcome,
    )


def evaluate_durable_finalization_guard(
    existing_guard: DecisionFinalizeGuardState[T] | None,
    *,
    key: DecisionFinalizationKey,
    requested_outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
) -> DecisionDurableFinalizationResult[T]:
    """Apply pure guard semantics for one atomic backend transaction."""
    guard_state = (
        existing_guard
        if existing_guard is not None
        else initial_decision_finalize_guard(key)
    )
    if guard_state.key != key:
        raise ValueError("existing finalization guard key does not match requested key")
    try:
        guard_result = guard_decision_finalization(guard_state, requested_outcome)
    except DecisionFinalizationConflictError:
        if existing_guard is None:
            raise RuntimeError("conflict requires existing persisted authoritative outcome")
        return DecisionDurableFinalizationResult(
            disposition=DecisionDurableFinalizationDisposition.CONFLICT,
            guard_state=existing_guard,
        )
    if guard_result.disposition is DecisionFinalizeDisposition.FIRST_FINALIZATION:
        return DecisionDurableFinalizationResult(
            disposition=DecisionDurableFinalizationDisposition.COMMITTED,
            guard_state=guard_result.state,
        )
    return DecisionDurableFinalizationResult(
        disposition=DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY,
        guard_state=guard_result.state,
    )
