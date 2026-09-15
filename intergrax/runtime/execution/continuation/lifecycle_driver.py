# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R1 — private Execution Engine continuation lifecycle progression commands."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService


class ExecutionContinuationLifecycleDriver:
    """Internal pause/wait facts — not a public platform contract."""

    __slots__ = ("_service",)

    def __init__(self, service: ExecutionContinuationService) -> None:
        self._service = service

    def record_execution_reached_safe_pause(
        self,
        continuation_id: str,
        *,
        execution_pause_established: bool,
    ) -> PendingExecutionContinuation:
        """``PAUSE_REQUESTED → PAUSED`` after canonical Execution is quiesced."""
        return self._service.establish_execution_paused(
            continuation_id,
            execution_pause_established=execution_pause_established,
        )

    def record_ready_for_human_resolution(
        self,
        continuation_id: str,
    ) -> PendingExecutionContinuation:
        """``PAUSED → WAITING_FOR_HUMAN`` when blocked work awaits human input."""
        return self._service.establish_waiting_for_human(continuation_id)


__all__ = ["ExecutionContinuationLifecycleDriver"]
