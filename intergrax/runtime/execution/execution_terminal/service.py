# © Artur Czarnecki. All rights reserved.

"""Execution terminal domain service (P0C-5 / P0C-6)."""

from __future__ import annotations

from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
    ExecutionTerminalStore,
)
from intergrax.contracts.execution_identity import RunId, validate_run_id
from intergrax.runtime.execution.execution_terminal.durability_policy import (
    DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    normalize_terminal_record,
    validate_terminal_run_id_consistency,
)
from intergrax.runtime.task.task_trace import utc_now_iso


class ExecutionTerminalService:
    """Canonical authority for durable terminal execution outcomes."""

    __slots__ = ("_store",)

    def __init__(self, store: ExecutionTerminalStore) -> None:
        self._store = store

    @property
    def store(self) -> ExecutionTerminalStore:
        return self._store

    def require_durable(self) -> None:
        if not self._store.is_durable:
            raise ExecutionTerminalError(DURABLE_EXECUTION_TERMINAL_REQUIRED_MSG)

    def get_terminal_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        record = self._store.load_record(tenant_id=tenant_id, task_id=task_id)
        if record is None:
            return None
        return normalize_terminal_record(record)

    def is_execution_resumable(self, *, tenant_id: str, task_id: str) -> bool:
        try:
            record = self.get_terminal_record(tenant_id=tenant_id, task_id=task_id)
        except ExecutionTerminalError:
            return False
        return record is None

    def commit_terminal_outcome(
        self,
        *,
        tenant_id: str,
        task_id: str,
        outcome: ExecutionTerminalOutcome,
        run_id: RunId | None = None,
        reason: str = "",
        production_mode: bool = False,
    ) -> ExecutionTerminalRecord:
        if production_mode:
            self.require_durable()
        validated_run_id = validate_run_id(run_id) if run_id is not None else None
        bounded_reason = reason.strip()[:512]
        existing = self.get_terminal_record(tenant_id=tenant_id, task_id=task_id)
        if existing is not None:
            validate_terminal_run_id_consistency(existing, validated_run_id)
            if existing.outcome is outcome:
                return existing
            raise ExecutionTerminalConflictError(
                "execution terminal conflict: task already has a different terminal outcome",
                existing_outcome=existing.outcome,
            )
        record = ExecutionTerminalRecord(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=validated_run_id,
            outcome=outcome,
            reason=bounded_reason,
            recorded_at_utc=utc_now_iso(),
        )
        normalized = normalize_terminal_record(record)
        if not self._store.put_if_absent(normalized):
            raced = self.get_terminal_record(tenant_id=tenant_id, task_id=task_id)
            if raced is None:
                raise ExecutionTerminalError("execution terminal outcome race lost")
            validate_terminal_run_id_consistency(raced, validated_run_id)
            if raced.outcome is outcome:
                return raced
            raise ExecutionTerminalConflictError(
                "execution terminal conflict: task already has a different terminal outcome",
                existing_outcome=raced.outcome,
            )
        return normalized

    def record_cancellation(
        self,
        *,
        tenant_id: str,
        task_id: str,
        run_id: RunId | None = None,
        reason: str = "",
        production_mode: bool = False,
    ) -> ExecutionTerminalRecord:
        return self.commit_terminal_outcome(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            outcome=ExecutionTerminalOutcome.CANCELLED,
            reason=reason,
            production_mode=production_mode,
        )
