# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional Decision Lifecycle host seam hosted by canonical Execution (DS-EXEC-01)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.decision_identity import DecisionIdentity
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.runtime.decision_lifecycle_observability import DecisionLifecycleObserver


class DecisionLifecycleHost(Protocol):
    """Typed access to canonical Decision Lifecycle operations."""

    def start(
        self,
        identity: DecisionIdentity,
    ) -> DecisionLifecycleState:
        ...

    def transition(
        self,
        state: DecisionLifecycleState,
        to_stage: DecisionLifecycleStage,
    ) -> DecisionLifecycleState:
        ...


@dataclass(frozen=True, slots=True)
class CanonicalDecisionLifecycleHost:
    """Stateless host delegating 1:1 to canonical lifecycle contracts."""

    observer: DecisionLifecycleObserver | None = None

    def start(
        self,
        identity: DecisionIdentity,
    ) -> DecisionLifecycleState:
        state = initial_decision_lifecycle_state(identity)
        if self.observer is not None:
            self.observer.lifecycle_started(state)
        return state

    def transition(
        self,
        state: DecisionLifecycleState,
        to_stage: DecisionLifecycleStage,
    ) -> DecisionLifecycleState:
        new_state = transition_decision_lifecycle(state, to_stage)
        if self.observer is not None:
            self.observer.lifecycle_transitioned(
                previous_state=state,
                new_state=new_state,
            )
            if to_stage is DecisionLifecycleStage.TERMINAL:
                self.observer.lifecycle_terminal(new_state)
        return new_state
