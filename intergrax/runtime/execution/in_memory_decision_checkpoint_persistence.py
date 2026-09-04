# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference Decision checkpoint persistence (DS-CORE-06)."""

from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import DecisionFinalizationKey

T = TypeVar("T")


class InMemoryDecisionCheckpointPersistence(Generic[T]):
    """Reference checkpoint store for unit and conformance testing."""

    __slots__ = ("_lock", "_store")

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[DecisionFinalizationKey, DecisionCheckpointState[T]] = {}

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[T] | None:
        with self._lock:
            return self._store.get(key)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[T],
    ) -> None:
        with self._lock:
            self._store[checkpoint.finalization.key] = checkpoint
