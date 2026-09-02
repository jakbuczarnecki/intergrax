# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative worker lifecycle transition service (AW-2B)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityNotFound,
    WorkerInstanceRepository,
)
from intergrax.contracts.autonomous_work._validation import require_non_empty_text
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId, validate_worker_instance_id
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.autonomous_work.worker import WorkerInstance

_CANONICAL_ALLOWED_TRANSITIONS: Final[
    dict[WorkerLifecycleState, frozenset[WorkerLifecycleState]]
] = {
    WorkerLifecycleState.PROVISIONING: frozenset(
        {
            WorkerLifecycleState.ACTIVE,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.ACTIVE: frozenset(
        {
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.IDLE: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WORKING: frozenset(
        {
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WAITING_EXTERNAL: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.WAITING_FOR_HUMAN: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.RECOVERING: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.DEGRADED,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.DEGRADED: frozenset(
        {
            WorkerLifecycleState.WORKING,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.RECOVERING,
            WorkerLifecycleState.WAITING_EXTERNAL,
            WorkerLifecycleState.WAITING_FOR_HUMAN,
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.QUARANTINED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.PAUSED: frozenset(
        {
            WorkerLifecycleState.ACTIVE,
            WorkerLifecycleState.IDLE,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.QUARANTINED: frozenset(
        {
            WorkerLifecycleState.PAUSED,
            WorkerLifecycleState.STOPPED,
        }
    ),
    WorkerLifecycleState.STOPPED: frozenset(),
}


@runtime_checkable
class AutonomousWorkClock(Protocol):
    """Injectable UTC clock for deterministic lifecycle timestamps."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        ...


class AutonomousWorkInvalidLifecycleTransition(Exception):
    """Requested lifecycle edge is not allowed by canonical policy."""

    def __init__(
        self,
        message: str,
        *,
        worker_instance_id: WorkerInstanceId,
        from_state: WorkerLifecycleState,
        to_state: WorkerLifecycleState,
    ) -> None:
        super().__init__(message)
        self.worker_instance_id = worker_instance_id
        self.from_state = from_state
        self.to_state = to_state


class AutonomousWorkLifecycleStateConflict(Exception):
    """Caller expected a different persisted lifecycle state."""

    def __init__(
        self,
        message: str,
        *,
        worker_instance_id: WorkerInstanceId,
        expected_state: WorkerLifecycleState,
        actual_state: WorkerLifecycleState,
    ) -> None:
        super().__init__(message)
        self.worker_instance_id = worker_instance_id
        self.expected_state = expected_state
        self.actual_state = actual_state


class AutonomousWorkLifecycleClockError(Exception):
    """Clock produced a timestamp that would move updated_at backwards."""

    def __init__(
        self,
        message: str,
        *,
        worker_instance_id: WorkerInstanceId,
        previous_updated_at: datetime,
        attempted_updated_at: datetime,
    ) -> None:
        super().__init__(message)
        self.worker_instance_id = worker_instance_id
        self.previous_updated_at = previous_updated_at
        self.attempted_updated_at = attempted_updated_at


@dataclass(frozen=True, slots=True)
class WorkerLifecycleTransitionRequest:
    """Minimal governed transition request without authority credentials."""

    worker_instance_id: WorkerInstanceId
    expected_revision: Revision
    expected_state: WorkerLifecycleState
    target_state: WorkerLifecycleState
    transition_reason: str
    requested_at: datetime

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.expected_revision) is not Revision:
            raise TypeError("expected_revision must be Revision")
        validate_revision(self.expected_revision)
        if type(self.expected_state) is not WorkerLifecycleState:
            raise TypeError("expected_state must be WorkerLifecycleState")
        if type(self.target_state) is not WorkerLifecycleState:
            raise TypeError("target_state must be WorkerLifecycleState")
        object.__setattr__(
            self,
            "transition_reason",
            require_non_empty_text(
                self.transition_reason,
                label="transition_reason",
            ),
        )
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class WorkerLifecycleTransitionResult:
    """Outcome of an authoritative lifecycle transition attempt."""

    previous_state: WorkerLifecycleState
    current_state: WorkerLifecycleState
    worker_instance: WorkerInstance
    changed: bool


class WorkerLifecycleTransitionPolicy:
    """Pure canonical allow-list policy for Worker lifecycle edges."""

    __slots__ = ("_allowed_transitions",)

    def __init__(
        self,
        allowed_transitions: (
            dict[WorkerLifecycleState, frozenset[WorkerLifecycleState]] | None
        ) = None,
    ) -> None:
        source = (
            _CANONICAL_ALLOWED_TRANSITIONS
            if allowed_transitions is None
            else allowed_transitions
        )
        self._allowed_transitions = {
            state: frozenset(targets) for state, targets in source.items()
        }

    def allowed_targets(self, from_state: WorkerLifecycleState) -> frozenset[WorkerLifecycleState]:
        """Return the immutable allow-list for ``from_state``."""
        return self._allowed_transitions[from_state]

    def can_transition(
        self,
        from_state: WorkerLifecycleState,
        to_state: WorkerLifecycleState,
    ) -> bool:
        """Return whether ``to_state`` is an allowed target from ``from_state``."""
        if from_state == to_state:
            return False
        return to_state in self._allowed_transitions[from_state]

    def validate_transition(
        self,
        from_state: WorkerLifecycleState,
        to_state: WorkerLifecycleState,
        *,
        worker_instance_id: WorkerInstanceId,
    ) -> None:
        """Raise when the requested edge is not allowed."""
        if from_state == to_state:
            return
        if self.can_transition(from_state, to_state):
            return
        raise AutonomousWorkInvalidLifecycleTransition(
            (
                f"invalid worker lifecycle transition for {worker_instance_id}: "
                f"{from_state.value} -> {to_state.value}"
            ),
            worker_instance_id=worker_instance_id,
            from_state=from_state,
            to_state=to_state,
        )


class WorkerLifecycleService:
    """Authoritative semantic owner of Worker lifecycle transitions."""

    def __init__(
        self,
        *,
        repository: WorkerInstanceRepository,
        clock: AutonomousWorkClock | Callable[[], datetime],
        policy: WorkerLifecycleTransitionPolicy | None = None,
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._policy = policy or WorkerLifecycleTransitionPolicy()

    def get_current(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance:
        """Return the persisted worker instance for restart-safe rehydration."""
        validate_worker_instance_id(worker_instance_id)
        worker = self._repository.get(worker_instance_id=worker_instance_id)
        if worker is None:
            raise AutonomousWorkEntityNotFound(
                f"WorkerInstance not found for {worker_instance_id}"
            )
        return worker

    def transition(
        self,
        request: WorkerLifecycleTransitionRequest,
    ) -> WorkerLifecycleTransitionResult:
        """Validate and persist a canonical lifecycle transition."""
        worker = self.get_current(worker_instance_id=request.worker_instance_id)
        previous_state = worker.lifecycle_state

        if worker.lifecycle_state != request.expected_state:
            raise AutonomousWorkLifecycleStateConflict(
                (
                    f"worker lifecycle state conflict for {request.worker_instance_id}: "
                    f"expected {request.expected_state.value}, "
                    f"actual {worker.lifecycle_state.value}"
                ),
                worker_instance_id=request.worker_instance_id,
                expected_state=request.expected_state,
                actual_state=worker.lifecycle_state,
            )

        if request.target_state == previous_state:
            return WorkerLifecycleTransitionResult(
                previous_state=previous_state,
                current_state=previous_state,
                worker_instance=worker,
                changed=False,
            )

        self._policy.validate_transition(
            previous_state,
            request.target_state,
            worker_instance_id=request.worker_instance_id,
        )

        transition_at = self._now()
        if transition_at.tzinfo is None:
            raise AutonomousWorkLifecycleClockError(
                (
                    f"lifecycle clock produced naive timestamp for "
                    f"{request.worker_instance_id}"
                ),
                worker_instance_id=request.worker_instance_id,
                previous_updated_at=worker.updated_at,
                attempted_updated_at=transition_at,
            )
        if transition_at < worker.updated_at:
            raise AutonomousWorkLifecycleClockError(
                (
                    f"lifecycle clock moved updated_at backwards for "
                    f"{request.worker_instance_id}"
                ),
                worker_instance_id=request.worker_instance_id,
                previous_updated_at=worker.updated_at,
                attempted_updated_at=transition_at,
            )

        candidate = replace(
            worker,
            lifecycle_state=request.target_state,
            updated_at=transition_at,
        )
        persisted = self._repository.replace(
            candidate,
            expected_revision=request.expected_revision,
        )
        return WorkerLifecycleTransitionResult(
            previous_state=previous_state,
            current_state=persisted.lifecycle_state,
            worker_instance=persisted,
            changed=True,
        )

    def _now(self) -> datetime:
        if isinstance(self._clock, AutonomousWorkClock):
            return self._clock.now()
        return self._clock()
