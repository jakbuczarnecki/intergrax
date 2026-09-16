# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — in-memory execution continuation state store (non-durable default)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
    execution_continuation_lifecycle_is_terminal,
    execution_continuation_lifecycle_permits_successor_episode,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore


def _identity_key(identity: ExecutionContinuationIdentity) -> tuple[str, str, str, str]:
    return (
        str(identity.task_id),
        str(identity.run_id),
        str(identity.attempt_id),
        str(identity.execution_id),
    )


def _identity_matches(
    pending: PendingExecutionContinuation,
    identity: ExecutionContinuationIdentity,
) -> bool:
    return (
        pending.identity.task_id == identity.task_id
        and pending.identity.run_id == identity.run_id
        and pending.identity.attempt_id == identity.attempt_id
        and pending.identity.execution_id == identity.execution_id
    )


class InMemoryExecutionContinuationStateStore(ExecutionContinuationStateStore):
    """Process-local continuation store with real compare-and-swap under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, PendingExecutionContinuation] = {}
        self._current_by_identity: dict[tuple[str, str, str, str], str] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
        with self._lock:
            return self._by_id.get(continuation_id)

    def find_by_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._lock:
            matches = [
                pending
                for pending in self._by_id.values()
                if _identity_matches(pending, identity)
            ]
        if len(matches) != 1:
            return None
        return matches[0]

    def resolve_current_episode_for_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._lock:
            return self._resolve_current_episode_locked(identity)

    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return self.resolve_current_episode_for_identity(identity)

    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        with self._lock:
            if pending.continuation_id in self._by_id:
                return False
            active = self._active_for_identity_locked(pending.identity)
            if len(active) > 1:
                raise ExecutionContinuationError(
                    "multiple active continuation episodes for identity",
                    code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
                )
            key = _identity_key(pending.identity)
            current_id = self._current_by_identity.get(key)
            if current_id is not None:
                predecessor = self._by_id.get(current_id)
                if predecessor is None or not _identity_matches(predecessor, pending.identity):
                    raise ExecutionContinuationError(
                        "current continuation pointer inconsistent with identity",
                        code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
                    )
                if not execution_continuation_lifecycle_permits_successor_episode(
                    predecessor.lifecycle_state,
                ):
                    raise ExecutionContinuationError(
                        "current continuation episode does not permit successor",
                        code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
                    )
            elif active:
                raise ExecutionContinuationError(
                    "active continuation episode without current pointer",
                    code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
                )
            self._by_id[pending.continuation_id] = pending
            self._current_by_identity[key] = pending.continuation_id
            return True

    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        with self._lock:
            if pending.continuation_id in self._by_id:
                return False
            self._by_id[pending.continuation_id] = pending
            return True

    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        if continuation_id != expected.continuation_id or continuation_id != updated.continuation_id:
            return False
        with self._lock:
            current = self._by_id.get(continuation_id)
            if current is None or current != expected:
                return False
            self._by_id[continuation_id] = updated
            return True

    def _active_for_identity_locked(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> list[PendingExecutionContinuation]:
        return [
            pending
            for pending in self._by_id.values()
            if _identity_matches(pending, identity)
            and not execution_continuation_lifecycle_is_terminal(pending.lifecycle_state)
        ]

    def _resolve_current_episode_locked(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        active = self._active_for_identity_locked(identity)
        if len(active) > 1:
            raise ExecutionContinuationError(
                "multiple active continuation episodes for identity",
                code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
            )
        key = _identity_key(identity)
        current_id = self._current_by_identity.get(key)
        if current_id is not None:
            current = self._by_id.get(current_id)
            if current is None or not _identity_matches(current, identity):
                raise ExecutionContinuationError(
                    "current continuation pointer inconsistent with identity",
                    code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
                )
            if len(active) == 1 and active[0].continuation_id != current_id:
                raise ExecutionContinuationError(
                    "active continuation episode disagrees with current pointer",
                    code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
                )
            return current
        if len(active) == 1:
            return active[0]
        matches = [
            pending
            for pending in self._by_id.values()
            if _identity_matches(pending, identity)
        ]
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        raise ExecutionContinuationError(
            "ambiguous continuation identity without current episode pointer",
            code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
        )


def default_execution_continuation_state_store() -> InMemoryExecutionContinuationStateStore:
    """Process-local default for explicit continuation composition (not restart-safe)."""
    return InMemoryExecutionContinuationStateStore()


def wire_execution_continuation_state_store(
    *,
    state_store: ExecutionContinuationStateStore | None = None,
) -> ExecutionContinuationStateStore:
    """Dedicated continuation composition: ``None`` selects in-memory default store."""
    if state_store is not None:
        return state_store
    return default_execution_continuation_state_store()


__all__ = [
    "InMemoryExecutionContinuationStateStore",
    "default_execution_continuation_state_store",
    "wire_execution_continuation_state_store",
]
