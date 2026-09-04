# © Artur Czarnecki. All rights reserved.

"""Execution terminal persistence adapters (P0C-5)."""

from __future__ import annotations

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
from intergrax.runtime.task.task_state import TaskState

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
