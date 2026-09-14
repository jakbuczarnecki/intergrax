# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Structured correlation identities for ERL reliability diagnostics — never collapsed."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ReliabilityDiagnosticCorrelation(BaseModel):
    """
    Distinct identity domains for one diagnostic emission.

    Each field retains its own meaning; callers must not encode multiple identities into one string.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=256)
    reliability_case_id: str = Field(min_length=1, max_length=256)
    external_effect_contract_id: str = Field(min_length=1, max_length=256)
    execution_id: ExecutionId | None = None
    run_id: RunId | None = None
    task_id: TaskId | None = None
    attempt_id: AttemptId | None = None
    trace_id: str | None = Field(default=None, max_length=256)
    idempotency_key: str | None = Field(default=None, max_length=256)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId | None:
        if value is None:
            return None
        return validate_run_id(value)

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId | None:
        if value is None:
            return None
        return validate_task_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

__all__ = ["ReliabilityDiagnosticCorrelation"]
