# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application lifecycle state machine (APP-HOST-2A)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
)
from intergrax.hosting.contracts.public_data import validate_bounded_identifier
from intergrax.hosting.errors import HostedApplicationLifecycleTransitionError

_VALID_TRANSITIONS: dict[
    HostedApplicationLifecycleState,
    frozenset[HostedApplicationLifecycleState],
] = {
    HostedApplicationLifecycleState.CREATED: frozenset({HostedApplicationLifecycleState.STARTING}),
    HostedApplicationLifecycleState.STARTING: frozenset(
        {
            HostedApplicationLifecycleState.READY,
            HostedApplicationLifecycleState.STOPPING,
            HostedApplicationLifecycleState.FAILED,
        }
    ),
    HostedApplicationLifecycleState.READY: frozenset(
        {
            HostedApplicationLifecycleState.STOPPING,
            HostedApplicationLifecycleState.FAILED,
        }
    ),
    HostedApplicationLifecycleState.STOPPING: frozenset(
        {
            HostedApplicationLifecycleState.STOPPED,
            HostedApplicationLifecycleState.FAILED,
        }
    ),
    HostedApplicationLifecycleState.STOPPED: frozenset(),
    HostedApplicationLifecycleState.FAILED: frozenset(),
}

_TERMINAL_STATES = frozenset(
    {
        HostedApplicationLifecycleState.STOPPED,
        HostedApplicationLifecycleState.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class LifecycleTransitionRecord:
    """Immutable lifecycle transition history entry."""

    from_state: HostedApplicationLifecycleState
    to_state: HostedApplicationLifecycleState
    occurred_at: datetime
    reason_code: str


class HostedApplicationLifecycleController:
    """Thread-safe lifecycle controller for one hosted application engine instance."""

    def __init__(self, clock: HostedApplicationClock) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._state = HostedApplicationLifecycleState.CREATED
        self._accepting_new_work = False
        self._shutdown_requested = False
        self._last_transition_at = clock.now()
        self._reason_code = ""
        self._history: list[LifecycleTransitionRecord] = []

    @property
    def state(self) -> HostedApplicationLifecycleState:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in _TERMINAL_STATES

    @property
    def accepting_new_work_flag(self) -> bool:
        with self._lock:
            return self._accepting_new_work

    @property
    def shutdown_requested_flag(self) -> bool:
        with self._lock:
            return self._shutdown_requested

    def set_accepting_new_work(self, value: bool) -> None:
        with self._lock:
            self._accepting_new_work = value

    def set_shutdown_requested(self, value: bool) -> None:
        with self._lock:
            self._shutdown_requested = value

    def transition_history(self) -> tuple[LifecycleTransitionRecord, ...]:
        with self._lock:
            return tuple(self._history)

    def snapshot(self) -> HostedApplicationLifecycleSnapshot:
        with self._lock:
            return HostedApplicationLifecycleSnapshot(
                state=self._state,
                accepting_new_work=self._accepting_new_work,
                shutdown_requested=self._shutdown_requested,
                last_transition_at=self._last_transition_at,
                reason_code=self._reason_code,
            )

    def transition_to(
        self,
        target: HostedApplicationLifecycleState,
        *,
        reason_code: str = "",
    ) -> None:
        safe_reason = validate_bounded_identifier(reason_code or "transition", field_name="reason_code")
        occurred_at = self._clock.now()
        if occurred_at.tzinfo is None:
            raise HostedApplicationLifecycleTransitionError(
                "lifecycle clock produced naive timestamp"
            )
        with self._lock:
            if self._state in _TERMINAL_STATES:
                raise HostedApplicationLifecycleTransitionError(
                    f"cannot transition from terminal state {self._state.value}"
                )
            allowed = _VALID_TRANSITIONS.get(self._state, frozenset())
            if target not in allowed:
                raise HostedApplicationLifecycleTransitionError(
                    f"cannot transition host lifecycle from {self._state.value} to {target.value}"
                )
            previous = self._state
            self._state = target
            self._last_transition_at = occurred_at
            self._reason_code = safe_reason
            self._history.append(
                LifecycleTransitionRecord(
                    from_state=previous,
                    to_state=target,
                    occurred_at=occurred_at,
                    reason_code=safe_reason,
                )
            )
