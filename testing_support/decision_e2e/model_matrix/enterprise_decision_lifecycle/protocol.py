# © Artur Czarnecki. All rights reserved.

"""Pluggable provider contracts for enterprise decision lifecycle (DS-E2E-15J-L7)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleEvent,
    DecisionLifecycleState,
)


class DecisionIdentityProvider(Protocol):
    """Issues stable decision identifiers."""

    def new_decision_id(self) -> str: ...


class DecisionClockProvider(Protocol):
    """Supplies timestamps for lifecycle records and events."""

    def now(self) -> datetime: ...


class DecisionStateTransitionProvider(Protocol):
    """Validates and authorizes lifecycle state transitions."""

    def assert_transition_allowed(
        self,
        *,
        from_state: DecisionLifecycleState,
        to_state: DecisionLifecycleState,
    ) -> None: ...


class DecisionAuditProvider(Protocol):
    """Records lifecycle events for audit and replay."""

    def record_lifecycle_event(self, event: DecisionLifecycleEvent) -> None: ...


__all__ = [
    "DecisionAuditProvider",
    "DecisionClockProvider",
    "DecisionIdentityProvider",
    "DecisionStateTransitionProvider",
]
