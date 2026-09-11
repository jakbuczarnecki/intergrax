# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference Decision checkpoint persistence (DS-CORE-06)."""

from __future__ import annotations

from threading import Lock
from typing import Generic, TypeVar

from intergrax.contracts.decision_checkpoint import DecisionCheckpointState
from intergrax.contracts.decision_finalization import DecisionFinalizationKey
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    StaleDecisionCheckpointWriteError,
)

T = TypeVar("T")


class InMemoryDecisionCheckpointPersistence(Generic[T]):
    """Reference checkpoint store for unit and conformance testing."""

    __slots__ = ("_lock", "_revisions", "_store")

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: dict[DecisionFinalizationKey, DecisionCheckpointState[T]] = {}
        self._revisions: dict[DecisionFinalizationKey, int] = {}

    def materialized_revision(self, *, key: DecisionFinalizationKey) -> int:
        with self._lock:
            return self._revisions.get(key, 0)

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
        expected_revision: int | None = None,
    ) -> None:
        key = checkpoint.finalization.key
        with self._lock:
            if expected_revision is None:
                if key not in self._store:
                    self._revisions[key] = 1
                self._store[key] = checkpoint
                return
            current = self._revisions.get(key, 0)
            if current != expected_revision:
                raise StaleDecisionCheckpointWriteError(
                    f"expected snapshot_revision={expected_revision}, actual={current}",
                )
            self._store[key] = checkpoint
            self._revisions[key] = expected_revision + 1
