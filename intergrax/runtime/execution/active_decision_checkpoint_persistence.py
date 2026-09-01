# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active Decision checkpoint persistence binding (DS-EXEC-02)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ActiveDecisionCheckpointPersistenceBinding(Generic[T]):
    """Typed execution-scoped access token for one checkpoint payload type."""

    persistence: DecisionCheckpointPersistence[T]

    @classmethod
    def for_persistence(
        cls,
        persistence: DecisionCheckpointPersistence[T],
    ) -> ActiveDecisionCheckpointPersistenceBinding[T]:
        """Anchor checkpoint payload type ``T`` before execution-scoped access."""
        return cls(persistence)

    def get_active(self) -> DecisionCheckpointPersistence[T] | None:
        binding = _active_decision_checkpoint_persistence.get()
        if binding is None:
            return None
        if binding.persistence is not self.persistence:
            return None
        return self.persistence

    def require_active(self) -> DecisionCheckpointPersistence[T]:
        binding = _active_decision_checkpoint_persistence.get()
        if binding is None:
            raise RuntimeError("active decision checkpoint persistence required")
        if binding.persistence is not self.persistence:
            raise RuntimeError(
                "active decision checkpoint persistence does not match this binding",
            )
        return self.persistence


_active_decision_checkpoint_persistence: ContextVar[
    ActiveDecisionCheckpointPersistenceBinding | None
] = ContextVar(
    "active_decision_checkpoint_persistence",
    default=None,
)


def bind_active_decision_checkpoint_persistence[T](
    persistence: DecisionCheckpointPersistence[T],
) -> Token:
    return _active_decision_checkpoint_persistence.set(
        ActiveDecisionCheckpointPersistenceBinding(persistence),
    )


def reset_active_decision_checkpoint_persistence(token: Token) -> None:
    _active_decision_checkpoint_persistence.reset(token)


def is_decision_checkpoint_persistence_active() -> bool:
    return _active_decision_checkpoint_persistence.get() is not None
