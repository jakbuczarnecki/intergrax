# © Artur Czarnecki. All rights reserved.

"""Execution terminal persistence adapters (P0C-5)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalPersistenceCapability,
    ExecutionTerminalRecord,
    ExecutionTerminalStore,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.task.task_state import TaskState

_KV_KEY_PREFIX = "execution_terminal"
_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.execution_terminal.v1"
_TERMINAL_SCHEMA_VERSION = 1

_SUPPORTED_TERMINAL_OUTCOMES = frozenset(ExecutionTerminalOutcome)

_TASK_STATE_TERMINAL_OUTCOMES: dict[TaskState, ExecutionTerminalOutcome] = {
    TaskState.COMPLETED: ExecutionTerminalOutcome.COMPLETED,
    TaskState.PARTIALLY_COMPLETED: ExecutionTerminalOutcome.COMPLETED,
    TaskState.FAILED: ExecutionTerminalOutcome.FAILED,
    TaskState.CANCELLED: ExecutionTerminalOutcome.CANCELLED,
}

_TASK_STATE_TERMINAL_REASONS: dict[TaskState, str] = {
    TaskState.COMPLETED: "completed",
    TaskState.PARTIALLY_COMPLETED: "partially_completed",
    TaskState.FAILED: "failed",
    TaskState.CANCELLED: "cancelled",
}


def terminal_outcome_from_task_state(state: TaskState) -> ExecutionTerminalOutcome | None:
    """Map runtime terminal ``TaskState`` values to durable terminal outcomes."""
    return _TASK_STATE_TERMINAL_OUTCOMES.get(state)


def terminal_reason_for_task_state(state: TaskState) -> str:
    """Bounded reason token for a terminal ``TaskState`` projection."""
    return _TASK_STATE_TERMINAL_REASONS.get(state, "terminal")


_TERMINAL_OUTCOME_TASK_STATES: dict[ExecutionTerminalOutcome, TaskState] = {
    ExecutionTerminalOutcome.COMPLETED: TaskState.COMPLETED,
    ExecutionTerminalOutcome.FAILED: TaskState.FAILED,
    ExecutionTerminalOutcome.CANCELLED: TaskState.CANCELLED,
}


def task_state_from_terminal_outcome(outcome: ExecutionTerminalOutcome) -> TaskState:
    """Map durable terminal outcome to canonical ``TaskState`` projection."""
    return _TERMINAL_OUTCOME_TASK_STATES[outcome]


def reconcile_task_state_with_terminal_outcome(
    current_state: TaskState,
    canonical_outcome: ExecutionTerminalOutcome,
) -> TaskState:
    """Align local terminal projection with durable canonical outcome.

    When ``current_state`` maps to the same durable outcome, preserve it
    (e.g. ``PARTIALLY_COMPLETED`` when canonical is ``COMPLETED``). When the
    durable outcome differs, project the canonical ``TaskState`` — this is
    reconciliation against durable authority, not a lifecycle transition.
    """
    current_outcome = terminal_outcome_from_task_state(current_state)
    if current_outcome is canonical_outcome:
        return current_state
    return task_state_from_terminal_outcome(canonical_outcome)


def validate_terminal_run_id_consistency(
    existing: ExecutionTerminalRecord,
    incoming_run_id: RunId | None,
) -> None:
    """Fail closed when durable and active run identities disagree."""
    if existing.run_id is not None and incoming_run_id is not None:
        if existing.run_id != incoming_run_id:
            raise ExecutionTerminalConflictError(
                "execution terminal conflict: run_id mismatch",
                existing_outcome=existing.outcome,
            )


@dataclass(frozen=True, slots=True)
class TerminalCommitResolution:
    """Canonical durable terminal authority resolved during Nexus finalization."""

    canonical_record: ExecutionTerminalRecord | None
    should_publish_terminal_event: bool


def _kv_storage_key(task_id: str) -> str:
    return f"{_KV_KEY_PREFIX}:{task_id}"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def encode_terminal_record(record: ExecutionTerminalRecord) -> bytes:
    payload = {
        "schema_version": _TERMINAL_SCHEMA_VERSION,
        "tenant_id": record.tenant_id,
        "task_id": record.task_id,
        "run_id": str(record.run_id) if record.run_id is not None else None,
        "outcome": record.outcome.value,
        "reason": record.reason,
        "recorded_at_utc": record.recorded_at_utc,
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_terminal_record(raw: bytes) -> ExecutionTerminalRecord:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionTerminalError("invalid execution terminal record encoding") from exc
    if not isinstance(payload, dict):
        raise ExecutionTerminalError("invalid execution terminal record payload")
    if payload.get("schema_version") != _TERMINAL_SCHEMA_VERSION:
        raise ExecutionTerminalError("unsupported execution terminal schema version")
    outcome_raw = payload.get("outcome")
    if not isinstance(outcome_raw, str):
        raise ExecutionTerminalError("invalid execution terminal outcome")
    try:
        outcome = ExecutionTerminalOutcome(outcome_raw)
    except ValueError as exc:
        raise ExecutionTerminalError("invalid execution terminal outcome") from exc
    run_id: RunId | None = None
    run_raw = payload.get("run_id")
    if run_raw is not None:
        if not isinstance(run_raw, str):
            raise ExecutionTerminalError("invalid execution terminal run_id")
        run_id = validate_run_id(run_raw)
    tenant_id = payload.get("tenant_id")
    task_id = payload.get("task_id")
    reason = payload.get("reason")
    recorded_at_utc = payload.get("recorded_at_utc")
    if (
        not isinstance(tenant_id, str)
        or not isinstance(task_id, str)
        or not isinstance(reason, str)
        or not isinstance(recorded_at_utc, str)
    ):
        raise ExecutionTerminalError("invalid execution terminal record fields")
    return normalize_terminal_record(
        ExecutionTerminalRecord(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            outcome=outcome,
            reason=reason,
            recorded_at_utc=recorded_at_utc,
        ),
    )


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


class KvExecutionTerminalStore(ExecutionTerminalStore):
    """DistributedKVStore-backed terminal execution authority."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    @property
    def is_durable(self) -> bool:
        return True

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        raw = self._kv_store.get(tenant_id=tenant_id, key=_kv_storage_key(task_id))
        if raw is None:
            return None
        return decode_terminal_record(raw)

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        normalized = normalize_terminal_record(record)
        return self._kv_store.compare_and_set(
            tenant_id=normalized.tenant_id,
            key=_kv_storage_key(normalized.task_id),
            expected=None,
            new_value=encode_terminal_record(normalized),
        )


class DocumentStoreExecutionTerminalStore(ExecutionTerminalStore):
    """ConditionalDocumentStore-backed terminal execution authority."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "execution terminal persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    @property
    def is_durable(self) -> bool:
        return True

    def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        record = self._document_store.get(_document_partition(tenant_id), task_id)
        if record is None:
            return None
        return _document_record_to_terminal(record)

    def put_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        normalized = normalize_terminal_record(record)
        document = DocumentRecord(
            partition_key=_document_partition(normalized.tenant_id),
            row_key=normalized.task_id,
            data={"terminal": encode_terminal_record(normalized).decode("utf-8")},
        )
        return self._document_store.put_if_absent(document)


def _document_record_to_terminal(record: DocumentRecord) -> ExecutionTerminalRecord:
    terminal = record.data.get("terminal")
    if not isinstance(terminal, str):
        raise ExecutionTerminalError("invalid execution terminal document record")
    return decode_terminal_record(terminal.encode("utf-8"))


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
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
    checkpoint_store: ExecutionTerminalPersistenceCapability | None = None,
) -> ExecutionTerminalStore:
    """Platform composition boundary: storage capability → terminal store."""
    provided = sum(
        capability is not None
        for capability in (kv_store, document_store, checkpoint_store)
    )
    if provided > 1:
        raise ValueError(
            "wire_execution_terminal_store accepts kv_store, document_store, or "
            "checkpoint_store, not multiple",
        )
    if kv_store is not None:
        return KvExecutionTerminalStore(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "execution terminal persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreExecutionTerminalStore(document_store)
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
    if record.outcome not in _SUPPORTED_TERMINAL_OUTCOMES:
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
