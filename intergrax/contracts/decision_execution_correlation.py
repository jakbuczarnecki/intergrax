# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Immutable Decision ↔ Execution correlation evidence (DIAG R4).

Correlation records relate decision facts to execution scope. They do not imply
causality, root cause, or diagnostic authority for the Decision System.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.decision_identity import (
    DecisionId,
    DecisionIdentity,
    validate_decision_id,
    validate_decision_tenant_id,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)


class DecisionExecutionCorrelationKind(StrEnum):
    """Why a decision identity is linked to an execution scope (context only)."""

    DECISION_BOUND_EXECUTION = "decision_bound_execution"
    DECISION_RETRY_ATTEMPT = "decision_retry_attempt"


class DecisionExecutionCorrelationIntegrityError(Exception):
    """Raised when correlation evidence violates tenant or identity invariants."""


@dataclass(frozen=True, slots=True)
class DecisionExecutionCorrelationRecord:
    """Immutable evidence mapping decision scope to execution scope."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    decision_id: DecisionId
    decision_attempt_id: AttemptId
    execution_id: ExecutionId | None
    correlation_kind: DecisionExecutionCorrelationKind
    created_at: datetime

    def __post_init__(self) -> None:
        validate_decision_tenant_id(self.tenant_id)
        validate_task_id(self.task_id)
        validate_run_id(self.run_id)
        validate_decision_id(self.decision_id)
        validate_attempt_id(self.decision_attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if type(self.correlation_kind) is not DecisionExecutionCorrelationKind:
            raise TypeError(
                "correlation_kind must be DecisionExecutionCorrelationKind",
            )
        if type(self.created_at) is not datetime:
            raise TypeError("created_at must be datetime")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


def correlation_record_from_decision_identity(
    identity: DecisionIdentity,
    *,
    correlation_kind: DecisionExecutionCorrelationKind,
    created_at: datetime,
) -> DecisionExecutionCorrelationRecord:
    """Build correlation evidence from canonical DecisionIdentity (no cross-system copy)."""
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    if type(correlation_kind) is not DecisionExecutionCorrelationKind:
        raise TypeError("correlation_kind must be DecisionExecutionCorrelationKind")
    if type(created_at) is not datetime or created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware datetime")
    return DecisionExecutionCorrelationRecord(
        tenant_id=identity.tenant_id,
        task_id=identity.execution.task_id,
        run_id=identity.execution.run_id,
        decision_id=identity.decision_id,
        decision_attempt_id=identity.execution.attempt_id,
        execution_id=identity.execution.execution_id,
        correlation_kind=correlation_kind,
        created_at=created_at,
    )


def validate_correlation_tenant_scope(
    record: DecisionExecutionCorrelationRecord,
    *,
    tenant_id: str,
) -> None:
    """Reject cross-tenant correlation reads or writes."""
    validate_decision_tenant_id(tenant_id)
    if record.tenant_id != tenant_id:
        raise DecisionExecutionCorrelationIntegrityError(
            "correlation record tenant_id does not match lookup tenant scope",
        )
