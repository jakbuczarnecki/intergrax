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
    """Domain-neutral execution-scoped carrier for one persistence port."""

    persistence: DecisionCheckpointPersistence[T]


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


def get_active_decision_checkpoint_persistence() -> (
    DecisionCheckpointPersistence | None
):
    binding = _active_decision_checkpoint_persistence.get()
    if binding is None:
        return None
    return binding.persistence


def require_active_decision_checkpoint_persistence() -> DecisionCheckpointPersistence:
    persistence = get_active_decision_checkpoint_persistence()
    if persistence is None:
        raise RuntimeError("active decision checkpoint persistence required")
    return persistence
