# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution-boundary failure evidence contracts (DIAG R2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

_MAX_SAFE_SUMMARY_LEN = 256
_MAX_FAILURE_CODE_LEN = 128


class ExecutionFailureKind(StrEnum):
    """Narrow execution-boundary failure kind (not platform retry taxonomy)."""

    DELEGATE_EXCEPTION = "delegate_exception"


class ExecutionFailureEvidenceRecordStatus(StrEnum):
    PERSISTED = "persisted"
    UNAVAILABLE = "unavailable"


class ExecutionFailureEvidenceRequest(BaseModel):
    """Immutable request to record durable execution failure evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    failure_kind: ExecutionFailureKind
    safe_summary: str = Field(max_length=_MAX_SAFE_SUMMARY_LEN)
    failure_code: str | None = Field(default=None, max_length=_MAX_FAILURE_CODE_LEN)

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")


class ExecutionFailureEvidenceRecordResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: ExecutionFailureEvidenceRecordStatus
    event_id: EventId | None = None


class ExecutionFailureEvidenceRecorder(Protocol):
    def record_failure(
        self,
        request: ExecutionFailureEvidenceRequest,
    ) -> ExecutionFailureEvidenceRecordResult: ...


def validate_execution_failure_evidence_request(
    request: ExecutionFailureEvidenceRequest,
) -> ExecutionFailureEvidenceRequest:
    return ExecutionFailureEvidenceRequest(
        tenant_id=request.tenant_id,
        task_id=validate_task_id(request.task_id),
        run_id=validate_run_id(request.run_id),
        attempt_id=validate_attempt_id(request.attempt_id),
        execution_id=validate_execution_id(request.execution_id),
        failure_kind=request.failure_kind,
        safe_summary=request.safe_summary,
        failure_code=request.failure_code,
    )


__all__ = [
    "ExecutionFailureEvidenceRecordResult",
    "ExecutionFailureEvidenceRecordStatus",
    "ExecutionFailureEvidenceRecorder",
    "ExecutionFailureEvidenceRequest",
    "ExecutionFailureKind",
    "validate_execution_failure_evidence_request",
]
