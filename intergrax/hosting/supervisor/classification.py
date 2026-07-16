# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Terminal exit classification (APP-HOST-5A)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.hosting.contracts.lifecycle import HostedApplicationLifecycleState
from intergrax.hosting.contracts.public_data import validate_bounded_identifier, validate_instance_id
from intergrax.hosting.engine.diagnostics import (
  HostedApplicationEngineTerminalResult,
  HostedApplicationFailurePhase,
)
from intergrax.hosting.errors import (
  HostedApplicationConfigurationError,
  HostedApplicationDefinitionError,
  HostedApplicationInstanceConflictError,
  HostedApplicationSupervisorError,
)
from intergrax.hosting.shutdown import (
  HostedApplicationShutdownExecutionSnapshot,
  HostedApplicationShutdownPhase,
  HostedApplicationShutdownPhaseOutcome,
)

_CRITICAL_CLEANUP_PHASES = frozenset(
  {
    HostedApplicationShutdownPhase.COMPONENT_STOP,
    HostedApplicationShutdownPhase.RUNTIME_STOP,
    HostedApplicationShutdownPhase.LEASE_RELEASE,
  }
)


def _has_critical_cleanup_failure(
  shutdown_execution: HostedApplicationShutdownExecutionSnapshot | None,
) -> bool:
  if shutdown_execution is None:
    return False
  return any(
    record.phase in _CRITICAL_CLEANUP_PHASES
    and record.outcome is HostedApplicationShutdownPhaseOutcome.FAILED
    for record in shutdown_execution.phase_records
  )


_HOSTING_EXCEPTION_TYPES = (
  HostedApplicationInstanceConflictError,
  HostedApplicationConfigurationError,
  HostedApplicationDefinitionError,
  HostedApplicationSupervisorError,
)


class HostedApplicationExitKind(str, Enum):
  CLEAN_STOP = "clean_stop"
  CONFIGURATION_ERROR = "configuration_error"
  INSTANCE_CONFLICT = "instance_conflict"
  STARTUP_FAILURE = "startup_failure"
  RUNTIME_FAILURE = "runtime_failure"
  RESTART_REQUESTED = "restart_requested"
  FORCED_TERMINATION = "forced_termination"
  SUPERVISOR_ERROR = "supervisor_error"


class HostedApplicationExitRecord(BaseModel):
  """Safe terminal exit record without raw exception data."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  exit_kind: HostedApplicationExitKind
  retryable: bool
  reason_code: str
  application_id: str
  instance_id: str
  profile_digest: str
  terminal_lifecycle_state: HostedApplicationLifecycleState
  failure_phase: HostedApplicationFailurePhase | None = None
  failure_category: str = ""
  shutdown_timed_out: bool = False
  shutdown_forced: bool = False
  occurred_at: datetime

  @field_validator("instance_id")
  @classmethod
  def _validate_instance_id(cls, value: str) -> str:
    return validate_instance_id(value)

  @field_validator("reason_code", "failure_category")
  @classmethod
  def _validate_reason_code(cls, value: str) -> str:
    if not value:
      return ""
    return validate_bounded_identifier(value, field_name="reason_code")

  @field_validator("occurred_at")
  @classmethod
  def _validate_occurred_at(cls, value: datetime) -> datetime:
    if value.tzinfo is None:
      raise ValueError("occurred_at must be timezone-aware")
    return value


def _find_hosting_exception(exc: BaseException, *, max_depth: int = 8) -> BaseException | None:
  current: BaseException | None = exc
  for _ in range(max_depth):
    if current is None:
      return None
    for hosting_type in _HOSTING_EXCEPTION_TYPES:
      if isinstance(current, hosting_type):
        return current
    current = current.__cause__
  return None


@dataclass(frozen=True, slots=True)
class HostedApplicationExitClassifier:
  """Deterministic classifier for terminal engine and supervisor outcomes."""

  restart_requested: bool = False

  def classify_exception(
    self,
    exc: BaseException,
    *,
    application_id: str,
    instance_id: str,
    profile_digest: str,
    occurred_at: datetime,
  ) -> HostedApplicationExitRecord:
    hosting_exc = _find_hosting_exception(exc) or exc
    if isinstance(hosting_exc, HostedApplicationInstanceConflictError):
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.INSTANCE_CONFLICT,
        retryable=False,
        reason_code="instance_conflict",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=HostedApplicationLifecycleState.FAILED,
        occurred_at=occurred_at,
      )
    if isinstance(hosting_exc, (HostedApplicationConfigurationError, HostedApplicationDefinitionError)):
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.CONFIGURATION_ERROR,
        retryable=False,
        reason_code="configuration_error",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=HostedApplicationLifecycleState.FAILED,
        occurred_at=occurred_at,
      )
    if isinstance(hosting_exc, HostedApplicationSupervisorError):
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.SUPERVISOR_ERROR,
        retryable=False,
        reason_code="supervisor_error",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=HostedApplicationLifecycleState.FAILED,
        occurred_at=occurred_at,
      )
    return HostedApplicationExitRecord(
      exit_kind=HostedApplicationExitKind.STARTUP_FAILURE,
      retryable=True,
      reason_code="startup_failure",
      application_id=application_id,
      instance_id=instance_id,
      profile_digest=profile_digest,
      terminal_lifecycle_state=HostedApplicationLifecycleState.FAILED,
      occurred_at=occurred_at,
    )

  def classify_terminal_result(
    self,
    result: HostedApplicationEngineTerminalResult,
    *,
    application_id: str,
    instance_id: str,
    profile_digest: str,
    occurred_at: datetime,
    shutdown_execution: HostedApplicationShutdownExecutionSnapshot | None = None,
  ) -> HostedApplicationExitRecord:
    terminal_state = result.terminal_state
    failure = result.diagnostics.current_failure
    shutdown_timed_out = shutdown_execution.timed_out if shutdown_execution else False
    shutdown_forced = shutdown_execution.forced if shutdown_execution else False
    critical_cleanup_failure = _has_critical_cleanup_failure(shutdown_execution)

    if shutdown_timed_out or shutdown_forced or critical_cleanup_failure:
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.FORCED_TERMINATION,
        retryable=False,
        reason_code="forced_termination",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=terminal_state,
        shutdown_timed_out=shutdown_timed_out,
        shutdown_forced=shutdown_forced,
        occurred_at=occurred_at,
      )

    if self.restart_requested and terminal_state is HostedApplicationLifecycleState.STOPPED:
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.RESTART_REQUESTED,
        retryable=True,
        reason_code=result.reason_code or "restart_requested",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=terminal_state,
        shutdown_timed_out=shutdown_timed_out,
        shutdown_forced=shutdown_forced,
        occurred_at=occurred_at,
      )

    if terminal_state is HostedApplicationLifecycleState.STOPPED:
      return HostedApplicationExitRecord(
        exit_kind=HostedApplicationExitKind.CLEAN_STOP,
        retryable=False,
        reason_code=result.reason_code or "clean_stop",
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=terminal_state,
        occurred_at=occurred_at,
      )

    if terminal_state is HostedApplicationLifecycleState.FAILED:
      phase = failure.phase if failure is not None else None
      startup_phases = {
        HostedApplicationFailurePhase.INSTANCE_ACQUIRE,
        HostedApplicationFailurePhase.BEFORE_START_HOOK,
        HostedApplicationFailurePhase.COMPONENT_START,
        HostedApplicationFailurePhase.RUNTIME_FACTORY,
        HostedApplicationFailurePhase.RUNTIME_START,
        HostedApplicationFailurePhase.BEFORE_READY_HOOK,
        HostedApplicationFailurePhase.HEALTH_EVALUATION,
      }
      if phase in startup_phases:
        kind = HostedApplicationExitKind.STARTUP_FAILURE
        retryable = True
        reason = failure.reason_code if failure is not None else "startup_failure"
      else:
        kind = HostedApplicationExitKind.RUNTIME_FAILURE
        retryable = True
        reason = failure.reason_code if failure is not None else "runtime_failure"
      return HostedApplicationExitRecord(
        exit_kind=kind,
        retryable=retryable,
        reason_code=reason,
        application_id=application_id,
        instance_id=instance_id,
        profile_digest=profile_digest,
        terminal_lifecycle_state=terminal_state,
        failure_phase=phase,
        failure_category=phase.value if phase is not None else "",
        occurred_at=occurred_at,
      )

    return HostedApplicationExitRecord(
      exit_kind=HostedApplicationExitKind.RUNTIME_FAILURE,
      retryable=True,
      reason_code=result.reason_code or "runtime_failure",
      application_id=application_id,
      instance_id=instance_id,
      profile_digest=profile_digest,
      terminal_lifecycle_state=terminal_state,
      occurred_at=occurred_at,
    )
