# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Append-only Decision↔Execution correlation evidence persistence (DIAG R4)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationIntegrityError,
    DecisionExecutionCorrelationRecord,
    validate_correlation_tenant_scope,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId

MAX_CORRELATION_QUERY_RESULTS = 64


class DecisionExecutionCorrelationPersistence(Protocol):
    """Immutable correlation evidence store — Decision System is fact authority."""

    def append(self, record: DecisionExecutionCorrelationRecord) -> None:
        """Persist one correlation evidence record (append-only)."""

    def query_by_execution_scope(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
        limit: int = MAX_CORRELATION_QUERY_RESULTS,
    ) -> tuple[DecisionExecutionCorrelationRecord, ...]:
        """Bounded lookup for related decisions at execution scope."""


class InMemoryDecisionExecutionCorrelationPersistence:
    """Test and qualification harness backend."""

    def __init__(self) -> None:
        self._records: list[DecisionExecutionCorrelationRecord] = []

    def append(self, record: DecisionExecutionCorrelationRecord) -> None:
        if type(record) is not DecisionExecutionCorrelationRecord:
            raise TypeError("record must be DecisionExecutionCorrelationRecord")
        self._records.append(record)

    def query_by_execution_scope(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
        limit: int = MAX_CORRELATION_QUERY_RESULTS,
    ) -> tuple[DecisionExecutionCorrelationRecord, ...]:
        if limit < 1 or limit > MAX_CORRELATION_QUERY_RESULTS:
            raise ValueError(
                f"limit must be between 1 and {MAX_CORRELATION_QUERY_RESULTS}",
            )
        matched: list[DecisionExecutionCorrelationRecord] = []
        for record in self._records:
            try:
                validate_correlation_tenant_scope(record, tenant_id=tenant_id)
            except DecisionExecutionCorrelationIntegrityError:
                continue
            if record.task_id != task_id or record.run_id != run_id:
                continue
            if attempt_id is not None and record.decision_attempt_id != attempt_id:
                continue
            if execution_id is not None and record.execution_id != execution_id:
                continue
            matched.append(record)
        matched.sort(key=lambda item: item.created_at)
        return tuple(matched[:limit])
