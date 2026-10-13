# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-DUR-01 — serialized seal-before-ACK durable continuation store façade (T1–T7)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
from typing import Literal, Protocol

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.structured_json_value import StructuredJsonObject
from intergrax.hosting.instance._native_lock import NativeFileLock, NativeFileLockError
from intergrax.runtime.execution.continuation.durable_state_file import (
    AmbiguousDurablePersistError,
    ExecutionContinuationDurableStateFilePersistence,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    _identity_matches,
    begin_current_episode_if_predecessor_allows_in_maps,
    build_durable_export_payload_from_maps,
    compare_and_swap_in_maps,
    insert_if_absent_in_maps,
    resolve_current_episode_in_maps,
    restore_backing_maps,
    restore_durable_continuation_backing,
    snapshot_backing_maps,
)

MissingContinuationFilePolicy = Literal["empty", "fail"]


class _PersistOutcome(Protocol):
    def __call__(self, payload: StructuredJsonObject) -> None: ...


class SerializedSealExecutionContinuationStateStore(ExecutionContinuationStateStore):
    """
    File-backed continuation store with ADR-02-R3 T1–T7 serialized seal-before-ACK.

    Process-crash consistency at rename-level atomicity only (no power-loss claim).
    """

    def __init__(
        self,
        *,
        backing: ExecutionContinuationDurableBacking,
        file_persistence: ExecutionContinuationDurableStateFilePersistence,
        writer_lock: NativeFileLock,
        persist_export_payload: _PersistOutcome | None = None,
    ) -> None:
        self._backing = backing
        self._file_persistence = file_persistence
        self._writer_lock = writer_lock
        self._persist_export_payload = (
            persist_export_payload
            if persist_export_payload is not None
            else file_persistence.atomic_write_export_payload
        )
        self._reconciliation_required = False
        self._closed = False
        self._close_guard = threading.Lock()

    @property
    def is_durable(self) -> bool:
        return True

    @property
    def reconciliation_required(self) -> bool:
        return self._reconciliation_required

    def close(self) -> None:
        """Release single-writer file lease (process shutdown)."""
        with self._close_guard:
            with self._backing._lock:
                if self._closed:
                    return
                self._closed = True
                self._writer_lock.close()

    def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
        with self._backing._lock:
            return self._backing._by_id.get(continuation_id)

    def find_by_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._backing._lock:
            matches = [
                pending
                for pending in self._backing._by_id.values()
                if _identity_matches(pending, identity)
            ]
        if len(matches) != 1:
            return None
        return matches[0]

    def resolve_current_episode_for_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._backing._lock:
            return resolve_current_episode_in_maps(
                self._backing._by_id,
                self._backing._current_by_identity,
                identity,
            )

    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return self.resolve_current_episode_for_identity(identity)

    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        return self._run_sealed_mutation(
            lambda: begin_current_episode_if_predecessor_allows_in_maps(
                self._backing._by_id,
                self._backing._current_by_identity,
                pending,
            ),
        )

    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        return self._run_sealed_mutation(
            lambda: insert_if_absent_in_maps(self._backing._by_id, pending),
        )

    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        return self._run_sealed_mutation(
            lambda: compare_and_swap_in_maps(
                self._backing._by_id,
                continuation_id=continuation_id,
                expected=expected,
                updated=updated,
            ),
        )

    def _assert_mutation_allowed(self) -> None:
        if self._closed:
            raise ExecutionContinuationError(
                "durable continuation store is closed",
                code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
            )
        if not self._writer_lock.held:
            raise ExecutionContinuationError(
                "durable continuation writer lease is not held",
                code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
            )
        if self._reconciliation_required:
            raise ExecutionContinuationError(
                "durable continuation store requires reconciliation before further mutations",
                code=ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED,
            )

    def _run_sealed_mutation(self, mutate: Callable[[], bool]) -> bool:
        with self._close_guard:
            lock = self._backing._lock
            lock.acquire()
            try:
                self._assert_mutation_allowed()
                snapshot = snapshot_backing_maps(
                    self._backing._by_id,
                    self._backing._current_by_identity,
                )
                applied = mutate()
                if not applied:
                    return False
                payload = build_durable_export_payload_from_maps(
                    self._backing._by_id,
                    self._backing._current_by_identity,
                )
                try:
                    self._persist_export_payload(payload)
                except AmbiguousDurablePersistError:
                    self._reconciliation_required = True
                    raise ExecutionContinuationError(
                        "durable continuation persist outcome is ambiguous",
                        code=ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED,
                    ) from None
                except ExecutionContinuationError:
                    restore_backing_maps(
                        self._backing._by_id,
                        self._backing._current_by_identity,
                        snapshot,
                    )
                    raise
                except OSError as exc:
                    restore_backing_maps(
                        self._backing._by_id,
                        self._backing._current_by_identity,
                        snapshot,
                    )
                    raise ExecutionContinuationError(
                        "failed to persist durable continuation state",
                        code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
                    ) from exc
                return True
            finally:
                lock.release()


def open_serialized_seal_execution_continuation_state_store(
    state_file: Path,
    *,
    missing_file_policy: MissingContinuationFilePolicy = "empty",
    persist_export_payload: _PersistOutcome | None = None,
) -> SerializedSealExecutionContinuationStateStore:
    """
    Construct a single-writer serialized-seal store for ``state_file``.

    Raises when another process holds the writer lock on the same path.
    """
    lock_path = state_file.with_suffix(state_file.suffix + ".writer.lock")
    try:
        writer_lock = NativeFileLock(lock_path)
    except NativeFileLockError as exc:
        raise ExecutionContinuationError(
            "conflicting durable continuation writer ownership",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc

    file_persistence = ExecutionContinuationDurableStateFilePersistence(state_file)
    try:
        if state_file.is_file():
            backing = file_persistence.restore_backing()
        elif missing_file_policy == "empty":
            backing = ExecutionContinuationDurableBacking()
        else:
            raise ExecutionContinuationError(
                "durable continuation state file is missing",
                code=ExecutionContinuationErrorCode.NOT_FOUND,
            )
    except Exception:
        writer_lock.close()
        raise

    return SerializedSealExecutionContinuationStateStore(
        backing=backing,
        file_persistence=file_persistence,
        writer_lock=writer_lock,
        persist_export_payload=persist_export_payload,
    )


def reload_serialized_seal_store_from_disk(
    store: SerializedSealExecutionContinuationStateStore,
) -> None:
    """Reload backing from the on-disk envelope (reconciliation path)."""
    lock = store._backing._lock
    lock.acquire()
    try:
        payload = store._file_persistence.load_export_payload()
        restored = restore_durable_continuation_backing(payload)
        store._backing._by_id.clear()
        store._backing._by_id.update(restored._by_id)
        store._backing._current_by_identity.clear()
        store._backing._current_by_identity.update(restored._current_by_identity)
        store._reconciliation_required = False
    finally:
        lock.release()


__all__ = [
    "AmbiguousDurablePersistError",
    "SerializedSealExecutionContinuationStateStore",
    "open_serialized_seal_execution_continuation_state_store",
    "reload_serialized_seal_store_from_disk",
]
