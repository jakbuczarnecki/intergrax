# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — in-memory execution continuation state store (non-durable default)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore


class InMemoryExecutionContinuationStateStore(ExecutionContinuationStateStore):
    """Process-local continuation store with real compare-and-swap under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, PendingExecutionContinuation] = {}

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
                if pending.identity.task_id == identity.task_id
                and pending.identity.run_id == identity.run_id
                and pending.identity.attempt_id == identity.attempt_id
                and pending.identity.execution_id == identity.execution_id
            ]
        if len(matches) != 1:
            return None
        return matches[0]

    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._lock:
            matches = [
                pending
                for pending in self._by_id.values()
                if pending.identity.task_id == identity.task_id
                and pending.identity.run_id == identity.run_id
                and pending.identity.attempt_id == identity.attempt_id
                and pending.identity.execution_id == identity.execution_id
            ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ExecutionContinuationError(
                "ambiguous continuation identity for execution progress",
                code=ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY,
            )
        return matches[0]

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


def wire_execution_continuation_state_store(
    *,
    state_store: ExecutionContinuationStateStore | None = None,
) -> ExecutionContinuationStateStore:
    """Platform composition boundary: optional injected store or in-memory default."""
    if state_store is not None:
        return state_store
    return InMemoryExecutionContinuationStateStore()


__all__ = [
    "InMemoryExecutionContinuationStateStore",
    "wire_execution_continuation_state_store",
]
