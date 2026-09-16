# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — canonical Execution Engine continuation lifecycle service."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationTransition,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    advance_continuation_lifecycle,
    apply_resolution_to_pending,
    apply_resume_to_pending,
    assert_execution_continuation_identity_match,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore


class ExecutionContinuationService:
    """Execution Engine capability implementing :class:`ExecutionContinuationPort`."""

    __slots__ = ("_store",)

    def __init__(self, store: ExecutionContinuationStateStore) -> None:
        self._store = store

    @property
    def store(self) -> ExecutionContinuationStateStore:
        return self._store

    def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
        state = advance_continuation_lifecycle(
            None,
            ExecutionContinuationTransition.REQUEST_PAUSE,
        )
        pending = PendingExecutionContinuation(
            continuation_id=request.continuation_id,
            identity=request.identity,
            lifecycle_state=state,
            revision=1,
            reason=request.reason,
            governed_correlation=request.governed_correlation,
            pause_id=request.pause_id,
            human_request_id=request.human_request_id,
            requested_at=request.requested_at,
        )
        if not self._store.begin_current_episode_if_predecessor_allows(pending):
            raise ExecutionContinuationError(
                "duplicate continuation",
                code=ExecutionContinuationErrorCode.DUPLICATE_CONTINUATION,
            )
        return pending

    def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        pending = self._resolve_lookup(lookup)
        if lookup.identity is not None:
            assert_execution_continuation_identity_match(lookup.identity, pending.identity)
        return pending

    def apply_resolution(
        self,
        command: ExecutionContinuationResolutionCommand,
    ) -> PendingExecutionContinuation:
        pending = self._require(command.continuation_id)
        try:
            updated = apply_resolution_to_pending(pending, command)
        except ExecutionContinuationError:
            raise
        if not self._store.compare_and_swap(
            continuation_id=command.continuation_id,
            expected=pending,
            updated=updated,
        ):
            raise ExecutionContinuationError(
                "stale continuation revision for resolution",
                code=ExecutionContinuationErrorCode.STALE_REVISION,
            )
        return updated

    def resume(self, command: ExecutionContinuationResumeCommand) -> PendingExecutionContinuation:
        pending = self._require(command.continuation_id)
        try:
            updated = apply_resume_to_pending(pending, command)
        except ExecutionContinuationError:
            raise
        if not self._store.compare_and_swap(
            continuation_id=command.continuation_id,
            expected=pending,
            updated=updated,
        ):
            raise ExecutionContinuationError(
                "stale continuation revision for resume",
                code=ExecutionContinuationErrorCode.STALE_REVISION,
            )
        return updated

    def establish_execution_paused(
        self,
        continuation_id: str,
        *,
        execution_pause_established: bool,
    ) -> PendingExecutionContinuation:
        """Internal command: ``PAUSE_REQUESTED → PAUSED`` after Execution quiescence."""
        if not execution_pause_established:
            raise ExecutionContinuationError(
                "execution pause not established",
                code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
            )
        pending = self._require(continuation_id)
        return self._commit_lifecycle_transition(
            pending,
            ExecutionContinuationTransition.ADVANCE_TO_PAUSED,
        )

    def establish_waiting_for_human(self, continuation_id: str) -> PendingExecutionContinuation:
        """Internal command: ``PAUSED → WAITING_FOR_HUMAN``."""
        pending = self._require(continuation_id)
        return self._commit_lifecycle_transition(
            pending,
            ExecutionContinuationTransition.ADVANCE_TO_HUMAN_WAIT,
        )

    def _resolve_lookup(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
        if lookup.continuation_id is not None:
            return self._require(lookup.continuation_id)
        if lookup.identity is None:
            raise ExecutionContinuationError(
                "continuation not found",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
        pending = self._store.resolve_current_episode_for_identity(lookup.identity)
        if pending is None:
            raise ExecutionContinuationError(
                "continuation not found",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
        return pending

    def _require(self, continuation_id: str) -> PendingExecutionContinuation:
        pending = self._store.load(continuation_id)
        if pending is None:
            raise ExecutionContinuationError(
                "continuation not found",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
        return pending

    def _commit_lifecycle_transition(
        self,
        pending: PendingExecutionContinuation,
        transition: ExecutionContinuationTransition,
    ) -> PendingExecutionContinuation:
        next_state = advance_continuation_lifecycle(pending.lifecycle_state, transition)
        updated = pending.model_copy(
            update={
                "lifecycle_state": next_state,
                "revision": pending.revision + 1,
            },
        )
        if not self._store.compare_and_swap(
            continuation_id=pending.continuation_id,
            expected=pending,
            updated=updated,
        ):
            raise ExecutionContinuationError(
                "stale continuation revision for lifecycle progression",
                code=ExecutionContinuationErrorCode.STALE_REVISION,
            )
        return updated


def execution_continuation_port(
    service: ExecutionContinuationService,
) -> ExecutionContinuationPort:
    """Narrow typed view for consumers that must depend on the port only."""
    return service


__all__ = ["ExecutionContinuationService", "execution_continuation_port"]
