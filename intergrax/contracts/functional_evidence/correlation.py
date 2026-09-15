# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution correlation for functional evidence scope (Evidence Plane contract)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

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


class FunctionalEvidenceExecutionCorrelation(BaseModel):
    """Five-ID execution scope for functional evidence facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id_field(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id_field(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id_field(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id_field(cls, value: object) -> ExecutionId:
        return validate_execution_id(value)

    @field_validator("tenant_id")
    @classmethod
    def _require_tenant_id(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("tenant_id must be str")
        normalized = value.strip()
        if not normalized:
            raise ValueError("tenant_id must be non-empty and not whitespace-only")
        if value != normalized:
            raise ValueError("tenant_id must not contain leading or trailing whitespace")
        return normalized


__all__ = ["FunctionalEvidenceExecutionCorrelation"]
