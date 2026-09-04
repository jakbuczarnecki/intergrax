# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active Decision finalization persistence binding (DS-EXEC-03)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ActiveDecisionFinalizationPersistenceBinding(Generic[T]):
    """Typed execution-scoped access token for one finalization payload type."""

    persistence: DecisionFinalizationPersistence[T]

    @classmethod
    def for_persistence(
        cls,
        persistence: DecisionFinalizationPersistence[T],
    ) -> ActiveDecisionFinalizationPersistenceBinding[T]:
        """Anchor finalization payload type ``T`` before execution-scoped access."""
        return cls(persistence)

    def get_active(self) -> DecisionFinalizationPersistence[T] | None:
        binding = _active_decision_finalization_persistence.get()
        if binding is None:
            return None
        if binding.persistence is not self.persistence:
            return None
        return self.persistence

    def require_active(self) -> DecisionFinalizationPersistence[T]:
        binding = _active_decision_finalization_persistence.get()
        if binding is None:
            raise RuntimeError("active decision finalization persistence required")
        if binding.persistence is not self.persistence:
            raise RuntimeError(
                "active decision finalization persistence does not match this binding",
            )
        return self.persistence


_active_decision_finalization_persistence: ContextVar[
    ActiveDecisionFinalizationPersistenceBinding | None
] = ContextVar(
    "active_decision_finalization_persistence",
    default=None,
)


def bind_active_decision_finalization_persistence[T](
    persistence: DecisionFinalizationPersistence[T],
) -> Token:
    return _active_decision_finalization_persistence.set(
        ActiveDecisionFinalizationPersistenceBinding(persistence),
    )


def reset_active_decision_finalization_persistence(token: Token) -> None:
    _active_decision_finalization_persistence.reset(token)


def is_decision_finalization_persistence_active() -> bool:
    return _active_decision_finalization_persistence.get() is not None
