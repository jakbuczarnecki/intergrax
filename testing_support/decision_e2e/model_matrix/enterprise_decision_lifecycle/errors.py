# © Artur Czarnecki. All rights reserved.

"""Controlled failures for enterprise decision lifecycle (DS-E2E-15J-L7)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleState,
)


class DecisionLifecycleError(Exception):
    """Base error for decision lifecycle boundary failures."""


class DecisionLifecycleTransitionRejectedError(DecisionLifecycleError):
    """Raised when a state transition is not permitted by the active provider."""

    def __init__(
        self,
        *,
        from_state: DecisionLifecycleState,
        to_state: DecisionLifecycleState,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Transition from {from_state.value} to {to_state.value} is not allowed"
        )


__all__ = [
    "DecisionLifecycleError",
    "DecisionLifecycleTransitionRejectedError",
]
