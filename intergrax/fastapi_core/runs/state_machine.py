# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.fastapi_core.runs.models import RunStatus


class InvalidRunTransitionError(Exception):
    """Raised when an illegal run status transition is attempted."""


class RunStateMachine:
    """
    Domain state machine defining legal run lifecycle transitions.
    """

    _allowed: dict[RunStatus, set[RunStatus]] = {
        RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELED},
        RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED},
        RunStatus.COMPLETED: set(),
        RunStatus.FAILED: set(),
        RunStatus.CANCELED: set(),
    }

    @classmethod
    def validate_transition(cls, current: RunStatus, target: RunStatus) -> None:
        if target not in cls._allowed[current]:
            raise InvalidRunTransitionError(
                f"Illegal run status transition: {current} → {target}"
            )
