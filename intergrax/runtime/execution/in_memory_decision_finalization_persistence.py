# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference Decision finalization persistence (DS-REC-01).

Deterministic single-process atomicity via lock — not production durability proof.
"""

from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar

from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    DecisionFinalizeGuardState,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
    DecisionDurableFinalizationResult,
    evaluate_durable_finalization_guard,
)

T = TypeVar("T")


class InMemoryDecisionFinalizationPersistence(Generic[T]):
    """Reference store simulating atomic finalization in one process."""

    __slots__ = ("_lock", "_store")

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[DecisionFinalizationKey, DecisionFinalizeGuardState[T]] = {}

    def load_guard_state(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionFinalizeGuardState[T] | None:
        with self._lock:
            return self._store.get(key)

    def commit_authoritative_outcome(
        self,
        *,
        key: DecisionFinalizationKey,
        requested_outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
    ) -> DecisionDurableFinalizationResult[T]:
        with self._lock:
            existing = self._store.get(key)
            result = evaluate_durable_finalization_guard(
                existing,
                key=key,
                requested_outcome=requested_outcome,
            )
            if result.disposition is DecisionDurableFinalizationDisposition.COMMITTED:
                self._store[key] = result.guard_state
            return result
