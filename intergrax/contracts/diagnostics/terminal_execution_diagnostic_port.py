# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Neutral terminal execution → diagnostic integration port (OBS-DIAG-PORT-1)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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


class TerminalDiagnosticDispatchStatus(StrEnum):
    """Transport-level outcome of one terminal diagnostic dispatch (not diagnostic domain)."""

    COMPLETED = "completed"
    FAILED_ISOLATED = "failed_isolated"
    SKIPPED = "skipped"


class TerminalDiagnosticDispatchResult(BaseModel):
    """Minimal acknowledgement for upstream execution integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TerminalDiagnosticDispatchStatus


class TerminalExecutionDiagnosticRequest(BaseModel):
    """
    One terminal execution fact submitted for downstream diagnostic interpretation.

    Diagnostic orchestration scope is **run-level** (tenant + task + run). Attempt and
    execution identifiers are an optional correlation pair (both absent or both present);
    partial correlation is invalid. They must not change run-scoped diagnostic grouping
    semantics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    observed_at: datetime
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("tenant_id must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("tenant_id must be non-empty")
        if normalized != value:
            raise ValueError("tenant_id must be canonical (no leading or trailing whitespace)")
        return value

    @field_validator("task_id", mode="before")
    @classmethod
    def _validate_task_id(cls, value: object) -> TaskId:
        return validate_task_id(value)

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object | None) -> AttemptId | None:
        if value is None:
            return None
        return validate_attempt_id(value)

    @field_validator("execution_id", mode="before")
    @classmethod
    def _validate_execution_id(cls, value: object | None) -> ExecutionId | None:
        if value is None:
            return None
        return validate_execution_id(value)

    @field_validator("observed_at")
    @classmethod
    def _validate_observed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_attempt_execution_pair(self) -> TerminalExecutionDiagnosticRequest:
        if (self.attempt_id is None) != (self.execution_id is None):
            raise ValueError(
                "attempt_id and execution_id must both be absent or both be present",
            )
        return self


@runtime_checkable
class TerminalExecutionDiagnosticPort(Protocol):
    """Pluggable consumer of terminal execution facts for diagnostic post-processing."""

    def dispatch_terminal_execution(
        self,
        request: TerminalExecutionDiagnosticRequest,
    ) -> TerminalDiagnosticDispatchResult | None: ...


__all__ = [
    "TerminalDiagnosticDispatchResult",
    "TerminalDiagnosticDispatchStatus",
    "TerminalExecutionDiagnosticPort",
    "TerminalExecutionDiagnosticRequest",
]
