# © Artur Czarnecki. All rights reserved.

"""Default pluggable providers for enterprise decision lifecycle (DS-E2E-15J-L7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleEvent,
    DecisionLifecycleState,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.errors import (
    DecisionLifecycleTransitionRejectedError,
)

_DEFAULT_ALLOWED: dict[DecisionLifecycleState, frozenset[DecisionLifecycleState]] = {
    DecisionLifecycleState.CREATED: frozenset({DecisionLifecycleState.EVALUATING}),
    DecisionLifecycleState.EVALUATING: frozenset(
        {
            DecisionLifecycleState.APPROVED,
            DecisionLifecycleState.REJECTED,
        }
    ),
    DecisionLifecycleState.APPROVED: frozenset({DecisionLifecycleState.EXECUTING}),
    DecisionLifecycleState.EXECUTING: frozenset(
        {
            DecisionLifecycleState.COMPLETED,
            DecisionLifecycleState.FAILED,
        }
    ),
    DecisionLifecycleState.REJECTED: frozenset(),
    DecisionLifecycleState.COMPLETED: frozenset(),
    DecisionLifecycleState.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DefaultDecisionStateTransitionProvider:
    """Standard enterprise decision lifecycle transition graph."""

    def assert_transition_allowed(
        self,
        *,
        from_state: DecisionLifecycleState,
        to_state: DecisionLifecycleState,
    ) -> None:
        allowed = _DEFAULT_ALLOWED.get(from_state, frozenset())
        if to_state not in allowed:
            raise DecisionLifecycleTransitionRejectedError(
                from_state=from_state,
                to_state=to_state,
            )


@dataclass(frozen=True, slots=True)
class UuidDecisionIdentityProvider:
    def new_decision_id(self) -> str:
        return str(uuid4())


@dataclass(frozen=True, slots=True)
class UtcDecisionClockProvider:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


@dataclass
class RecordingDecisionAuditProvider:
    """In-memory audit sink for tests and local diagnostics."""

    events: list[DecisionLifecycleEvent] = field(default_factory=list)

    def record_lifecycle_event(self, event: DecisionLifecycleEvent) -> None:
        self.events.append(event)


__all__ = [
    "DefaultDecisionStateTransitionProvider",
    "RecordingDecisionAuditProvider",
    "UtcDecisionClockProvider",
    "UuidDecisionIdentityProvider",
]
