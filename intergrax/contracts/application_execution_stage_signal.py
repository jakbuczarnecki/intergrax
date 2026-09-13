# © Artur Czarnecki. All rights reserved.

"""Neutral application execution stage signals for the RuntimeEvent spine (OBS-P1B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.event_severity import EventSeverity
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

_MAX_SUMMARY_LEN = 512
_MAX_OPAQUE_ID_LEN = 128


class ApplicationExecutionStageSignalError(ValueError):
    """Raised when a stage signal or correlation bundle is invalid."""


def _bounded_opaque(value: str, *, label: str) -> str:
    if type(value) is not str:
        raise ApplicationExecutionStageSignalError(f"{label} must be str")
    normalized = value.strip()
    if not normalized:
        raise ApplicationExecutionStageSignalError(f"{label} must be non-empty")
    if len(normalized) > _MAX_OPAQUE_ID_LEN:
        raise ApplicationExecutionStageSignalError(
            f"{label} exceeds maximum length {_MAX_OPAQUE_ID_LEN}",
        )
    return normalized


def _bounded_summary(value: str) -> str:
    if type(value) is not str:
        raise ApplicationExecutionStageSignalError("summary must be str")
    normalized = value.strip()
    if not normalized:
        raise ApplicationExecutionStageSignalError("summary must be non-empty")
    if len(normalized) > _MAX_SUMMARY_LEN:
        return normalized[: _MAX_SUMMARY_LEN - 3] + "..."
    return normalized


@dataclass(frozen=True, slots=True)
class ApplicationExecutionCorrelation:
    """
    Platform execution identity bound to a scenario-owned business run correlation id.

    ``scenario_execution_correlation_id`` carries scenario semantics (for example a
    pipeline run uuid). Canonical ``task_id`` / ``run_id`` / ``attempt_id`` /
    ``execution_id`` must be supplied by runtime composition or Execution Engine —
    they are not derived from scenario ids inside platform code.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    scenario_execution_correlation_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ApplicationExecutionStageSignalError("tenant_id must be non-empty")
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        object.__setattr__(
            self,
            "execution_id",
            validate_execution_id(self.execution_id),
        )
        object.__setattr__(
            self,
            "scenario_execution_correlation_id",
            _bounded_opaque(
                self.scenario_execution_correlation_id,
                label="scenario_execution_correlation_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class ApplicationExecutionStageSignal:
    """Redaction-safe, scenario-neutral stage fact for observability projection."""

    application_slug: str
    scenario_execution_correlation_id: str
    sequence: int
    stage_id: str
    event_category: str
    severity: EventSeverity
    summary: str
    outcome_status: str | None = None
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ApplicationExecutionStageSignalError("sequence must be a non-negative int")
        object.__setattr__(
            self,
            "application_slug",
            _bounded_opaque(self.application_slug, label="application_slug"),
        )
        object.__setattr__(
            self,
            "scenario_execution_correlation_id",
            _bounded_opaque(
                self.scenario_execution_correlation_id,
                label="scenario_execution_correlation_id",
            ),
        )
        object.__setattr__(
            self,
            "stage_id",
            _bounded_opaque(self.stage_id, label="stage_id"),
        )
        object.__setattr__(
            self,
            "event_category",
            _bounded_opaque(self.event_category, label="event_category"),
        )
        object.__setattr__(self, "summary", _bounded_summary(self.summary))
        if self.outcome_status is not None:
            object.__setattr__(
                self,
                "outcome_status",
                _bounded_opaque(self.outcome_status, label="outcome_status"),
            )
        if self.diagnostic_code is not None:
            object.__setattr__(
                self,
                "diagnostic_code",
                _bounded_opaque(self.diagnostic_code, label="diagnostic_code"),
            )


class ApplicationExecutionStageSignalEmitter(Protocol):
    """Replaceable projection sink into the canonical RuntimeEvent bus."""

    def emit(
        self,
        signal: ApplicationExecutionStageSignal,
        *,
        correlation: ApplicationExecutionCorrelation,
    ) -> None:
        """Project one stage signal; raises ``ApplicationExecutionStageSignalError`` on reject."""


__all__ = [
    "ApplicationExecutionCorrelation",
    "ApplicationExecutionStageSignal",
    "ApplicationExecutionStageSignalEmitter",
    "ApplicationExecutionStageSignalError",
]
