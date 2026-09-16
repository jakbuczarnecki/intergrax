# © Artur Czarnecki. All rights reserved.

"""GR-5-R2 — in-memory execution continuation state store (non-durable default)."""

from __future__ import annotations

import copy
import threading
from typing import Any

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
    execution_continuation_lifecycle_is_terminal,
    execution_continuation_lifecycle_permits_successor_episode,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)

DURABLE_CONTINUATION_STATE_SCHEMA_V1 = "execution_continuation_durable_state.v1"


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


class ExecutionContinuationDurableBacking:
    """External mutable state simulating durable persistence (reference qualification only)."""

    __slots__ = ("_lock", "_by_id", "_current_by_identity")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, PendingExecutionContinuation] = {}
        self._current_by_identity: dict[tuple[str, str, str, str], str] = {}


class BackingExecutionContinuationStateStore(InMemoryExecutionContinuationStateStore):
    """New store client over shared live backing (reconnect only, not restart-qualified)."""

    def __init__(self, backing: ExecutionContinuationDurableBacking) -> None:
        self._lock = backing._lock
        self._by_id = backing._by_id
        self._current_by_identity = backing._current_by_identity

    @property
    def is_durable(self) -> bool:
        return False


class ReconstructedDurableExecutionContinuationStateStore(BackingExecutionContinuationStateStore):
    """Store over backing reconstructed from serialized durable export (restart-qualified)."""

    @property
    def is_durable(self) -> bool:
        return True


def export_durable_continuation_state(
    backing: ExecutionContinuationDurableBacking,
) -> dict[str, Any]:
    """Export a deep, JSON-compatible durable snapshot (no mutable aliases to backing)."""
    with backing._lock:
        records = {
            continuation_id: pending.model_dump(mode="json")
            for continuation_id, pending in backing._by_id.items()
        }
        current_entries = [
            {
                "task_id": key[0],
                "run_id": key[1],
                "attempt_id": key[2],
                "execution_id": key[3],
                "continuation_id": continuation_id,
            }
            for key, continuation_id in backing._current_by_identity.items()
        ]
    return copy.deepcopy(
        {
            "schema_version": DURABLE_CONTINUATION_STATE_SCHEMA_V1,
            "records": records,
            "current_by_identity": current_entries,
        },
    )


def _identity_from_durable_key_fields(raw: dict[str, Any]) -> ExecutionContinuationIdentity:
    try:
        return ExecutionContinuationIdentity(
            task_id=TaskId(raw["task_id"]),
            run_id=RunId(raw["run_id"]),
            attempt_id=AttemptId(raw["attempt_id"]),
            execution_id=ExecutionId(raw["execution_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionContinuationError(
            "durable continuation export missing or invalid four-ID identity fields",
            code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
        ) from exc


def _validate_restored_store_invariants(
    by_id: dict[str, PendingExecutionContinuation],
    current_by_identity: dict[tuple[str, str, str, str], str],
) -> None:
    seen_identity_keys: set[tuple[str, str, str, str]] = set()
    for key, continuation_id in current_by_identity.items():
        if key in seen_identity_keys:
            raise ExecutionContinuationError(
                "duplicate current continuation pointer for identity in durable export",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        seen_identity_keys.add(key)
        current = by_id.get(continuation_id)
        if current is None:
            raise ExecutionContinuationError(
                "current continuation pointer references missing snapshot",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        if _identity_key(current.identity) != key:
            raise ExecutionContinuationError(
                "current continuation pointer identity mismatch",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )


def restore_durable_continuation_backing(payload: dict[str, Any]) -> ExecutionContinuationDurableBacking:
    """Construct a new backing from serialized durable state (new lock, new dicts)."""
    if payload.get("schema_version") != DURABLE_CONTINUATION_STATE_SCHEMA_V1:
        raise ExecutionContinuationError(
            "unknown durable continuation persistence schema",
            code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
        )
    raw_records = payload.get("records")
    raw_current = payload.get("current_by_identity")
    if not isinstance(raw_records, dict) or not isinstance(raw_current, list):
        raise ExecutionContinuationError(
            "malformed durable continuation persistence envelope",
            code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
        )
    by_id: dict[str, PendingExecutionContinuation] = {}
    for continuation_id, raw_pending in raw_records.items():
        if not isinstance(raw_pending, dict):
            raise ExecutionContinuationError(
                "continuation snapshot record is not an object",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        try:
            pending = PendingExecutionContinuation.model_validate(raw_pending)
        except Exception as exc:
            raise ExecutionContinuationError(
                "continuation snapshot failed validation during durable restore",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            ) from exc
        if pending.continuation_id != continuation_id:
            raise ExecutionContinuationError(
                "continuation snapshot id disagrees with durable record key",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        by_id[continuation_id] = pending
    current_by_identity: dict[tuple[str, str, str, str], str] = {}
    for entry in raw_current:
        if not isinstance(entry, dict):
            raise ExecutionContinuationError(
                "current pointer entry is not an object",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        identity = _identity_from_durable_key_fields(entry)
        continuation_id = entry.get("continuation_id")
        if not isinstance(continuation_id, str) or not continuation_id.strip():
            raise ExecutionContinuationError(
                "current pointer entry missing continuation_id",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        key = _identity_key(identity)
        if key in current_by_identity:
            raise ExecutionContinuationError(
                "duplicate current continuation pointer for identity in durable export",
                code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
            )
        current_by_identity[key] = continuation_id.strip()
    _validate_restored_store_invariants(by_id, current_by_identity)
    backing = ExecutionContinuationDurableBacking()
    backing._by_id = by_id
    backing._current_by_identity = current_by_identity
    return backing


def reconstructed_durable_execution_continuation_state_store(
    backing: ExecutionContinuationDurableBacking,
) -> ReconstructedDurableExecutionContinuationStateStore:
    """Restart-qualified store view over backing produced by :func:`restore_durable_continuation_backing`."""
    return ReconstructedDurableExecutionContinuationStateStore(backing)


def execution_continuation_state_store_from_durable_export(
    payload: dict[str, Any],
) -> ReconstructedDurableExecutionContinuationStateStore:
    """Deserialize durable export into a restart-qualified continuation store."""
    backing = restore_durable_continuation_backing(payload)
    return reconstructed_durable_execution_continuation_state_store(backing)


def backing_execution_continuation_state_store(
    backing: ExecutionContinuationDurableBacking,
) -> BackingExecutionContinuationStateStore:
    """Construct a new durable-capable store view over ``backing``."""
    return BackingExecutionContinuationStateStore(backing)


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
    "BackingExecutionContinuationStateStore",
    "DURABLE_CONTINUATION_STATE_SCHEMA_V1",
    "ExecutionContinuationDurableBacking",
    "InMemoryExecutionContinuationStateStore",
    "ReconstructedDurableExecutionContinuationStateStore",
    "backing_execution_continuation_state_store",
    "default_execution_continuation_state_store",
    "execution_continuation_state_store_from_durable_export",
    "export_durable_continuation_state",
    "reconstructed_durable_execution_continuation_state_store",
    "restore_durable_continuation_backing",
    "wire_execution_continuation_state_store",
]
