# © Artur Czarnecki. All rights reserved.

"""Execution terminal persistence adapters (P0C-5)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalPersistenceCapability,
    ExecutionTerminalRecord,
    ExecutionTerminalStore,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id


class InMemoryExecutionTerminalStore(ExecutionTerminalStore):
    """Process-local terminal store for tests and single-process hosts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str], ExecutionTerminalRecord] = {}

    @property
    def is_durable(self) -> bool:
        return False

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        key = (tenant_id, task_id)
        with self._lock:
            return self._records.get(key)

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        key = (record.tenant_id, record.task_id)
        with self._lock:
            if key in self._records:
                return False
            self._records[key] = record
            return True


class CheckpointStoreExecutionTerminalStore(ExecutionTerminalStore):
    """Reuse durable checkpoint storage for terminal cancellation authority."""

    def __init__(self, checkpoint_store: ExecutionTerminalPersistenceCapability) -> None:
        self._checkpoint_store = checkpoint_store

    @property
    def is_durable(self) -> bool:
        return True

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        return self._checkpoint_store.get_terminal_record(tenant_id=tenant_id, task_id=task_id)

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        return self._checkpoint_store.put_terminal_record_if_absent(record)


def wire_execution_terminal_store(
    *,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None = None,
) -> ExecutionTerminalStore:
    if checkpoint_store is not None and isinstance(
        checkpoint_store,
        ExecutionTerminalPersistenceCapability,
    ):
        return CheckpointStoreExecutionTerminalStore(checkpoint_store)
    return InMemoryExecutionTerminalStore()


def normalize_terminal_record(record: ExecutionTerminalRecord) -> ExecutionTerminalRecord:
    """Fail closed on malformed durable terminal authority."""
    tenant_id = record.tenant_id.strip()
    task_id = record.task_id.strip()
    if not tenant_id or not task_id:
        raise ExecutionTerminalError("invalid execution terminal record identity")
    if not record.recorded_at_utc.strip():
        raise ExecutionTerminalError("invalid execution terminal recorded_at_utc")
    if record.outcome is not ExecutionTerminalOutcome.CANCELLED:
        raise ExecutionTerminalError("unsupported execution terminal outcome")
    run_id: RunId | None = None
    if record.run_id is not None:
        run_id = validate_run_id(record.run_id)
    reason = record.reason.strip()
    if len(reason) > 512:
        raise ExecutionTerminalError("execution terminal reason exceeds max length")
    return ExecutionTerminalRecord(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        outcome=record.outcome,
        reason=reason,
        recorded_at_utc=record.recorded_at_utc,
    )
