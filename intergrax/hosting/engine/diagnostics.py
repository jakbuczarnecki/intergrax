# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Safe hosting diagnostics and failure records (APP-HOST-3C)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.hosting.contracts.context import HostedApplicationClock
from intergrax.hosting.contracts.lifecycle import (
    HostedApplicationLifecycleSnapshot,
    HostedApplicationLifecycleState,
)
from intergrax.hosting.contracts.public_data import validate_bounded_identifier
from intergrax.hosting.errors import HostedApplicationDiagnosticError
from intergrax.hosting.engine.health import HostedApplicationHealthSnapshot

FailureIdGenerator = Callable[[], str]


class HostedApplicationFailurePhase(str, Enum):
    DEFINITION = "definition"
    INSTANCE_ACQUIRE = "instance_acquire"
    BEFORE_START_HOOK = "before_start_hook"
    COMPONENT_START = "component_start"
    RUNTIME_FACTORY = "runtime_factory"
    RUNTIME_START = "runtime_start"
    AFTER_START_OBSERVER = "after_start_observer"
    BEFORE_READY_HOOK = "before_ready_hook"
    HEALTH_EVALUATION = "health_evaluation"
    RUNTIME_OPERATION = "runtime_operation"
    BEFORE_STOP_HOOK = "before_stop_hook"
    COMPONENT_STOP = "component_stop"
    RUNTIME_STOP = "runtime_stop"
    EVENT_PUBLISH = "event_publish"
    EVENT_SUBSCRIBER = "event_subscriber"
    AFTER_STOP_OBSERVER = "after_stop_observer"
    ROLLBACK = "rollback"


class HostedApplicationOperationPhase(str, Enum):
    """Explicit active engine operation phase for diagnostics."""

    IDLE = "idle"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ROLLBACK = "rollback"


class HostedApplicationFailureRecord(BaseModel):
    """Safe public failure record without raw exception data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_id: str
    reason_code: str
    phase: HostedApplicationFailurePhase
    source_kind: str
    source_id: str
    exception_type: str
    occurred_at: datetime
    primary: bool = True
    secondary_failure_count: int = 0


class HostedApplicationDiagnosticSnapshot(BaseModel):
    """Deterministic safe diagnostic snapshot for one engine instance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    instance_id: str
    profile_digest: str
    definition_digest: str
    lifecycle: HostedApplicationLifecycleSnapshot
    health: HostedApplicationHealthSnapshot
    runtime_created: bool
    runtime_started: bool
    started_component_ids: tuple[str, ...] = ()
    component_start_order: tuple[str, ...] = ()
    hook_ids_by_point: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    event_subscription_ids: tuple[str, ...] = ()
    current_failure: HostedApplicationFailureRecord | None = None
    last_failure: HostedApplicationFailureRecord | None = None
    secondary_failures: tuple[HostedApplicationFailureRecord, ...] = ()
    context_closed: bool = False
    instance_lease_acquired: bool = False
    instance_lease_released: bool = False
    observer_task_count: int = 0
    active_operation_phase: HostedApplicationOperationPhase = HostedApplicationOperationPhase.IDLE
    snapshot_timestamp: datetime


class HostedApplicationEngineTerminalResult(BaseModel):
    """Terminal engine result after shutdown completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    terminal_state: HostedApplicationLifecycleState
    reason_code: str = ""
    diagnostics: HostedApplicationDiagnosticSnapshot


@dataclass
class DiagnosticsRecorder:
    """Internal diagnostics recorder with safe public projections."""

    clock: HostedApplicationClock
    application_id: str
    instance_id: str
    profile_digest: str
    definition_digest: str
    component_start_order: tuple[str, ...] = ()
    hook_ids_by_point: dict[str, tuple[str, ...]] = field(default_factory=dict)
    event_subscription_ids: tuple[str, ...] = ()
    failure_id_generator: FailureIdGenerator | None = None
    runtime_created: bool = False
    runtime_started: bool = False
    started_component_ids: list[str] = field(default_factory=list)
    context_closed: bool = False
    instance_lease_acquired: bool = False
    instance_lease_released: bool = False
    observer_task_count: int = 0
    active_operation_phase: HostedApplicationOperationPhase = HostedApplicationOperationPhase.IDLE
    current_failure: HostedApplicationFailureRecord | None = None
    last_failure: HostedApplicationFailureRecord | None = None
    secondary_failures: list[HostedApplicationFailureRecord] = field(default_factory=list)
    _primary_exception: BaseException | None = field(default=None, repr=False)
    _failure_counter: int = field(default=0, repr=False)

    @property
    def primary_exception(self) -> BaseException | None:
        return self._primary_exception

    def set_operation_phase(self, phase: HostedApplicationOperationPhase) -> None:
        self.active_operation_phase = phase

    def mark_runtime_created(self) -> None:
        self.runtime_created = True

    def mark_runtime_started(self) -> None:
        self.runtime_started = True

    def mark_component_started(self, component_id: str) -> None:
        if component_id not in self.started_component_ids:
            self.started_component_ids.append(component_id)

    def mark_lease_acquired(self) -> None:
        self.instance_lease_acquired = True

    def mark_lease_released(self) -> None:
        self.instance_lease_released = True

    def mark_context_closed(self) -> None:
        self.context_closed = True

    def set_observer_task_count(self, count: int) -> None:
        self.observer_task_count = count

    def next_failure_id(self) -> str:
        if self.failure_id_generator is not None:
            raw_id = self.failure_id_generator()
        else:
            self._failure_counter += 1
            raw_id = f"failure-{self._failure_counter:04d}"
        return self._validate_failure_id(raw_id)

    def record_primary_failure(
        self,
        *,
        phase: HostedApplicationFailurePhase,
        source_kind: str,
        source_id: str,
        exc: BaseException,
        reason_code: str,
    ) -> HostedApplicationFailureRecord:
        record = self._build_failure_record(
            phase=phase,
            source_kind=source_kind,
            source_id=source_id,
            exc=exc,
            reason_code=reason_code,
            primary=True,
        )
        self.current_failure = record
        self.last_failure = record
        self._primary_exception = exc
        return record

    def record_secondary_failure(
        self,
        *,
        phase: HostedApplicationFailurePhase,
        source_kind: str,
        source_id: str,
        exc: BaseException,
        reason_code: str = "secondary_failure",
    ) -> HostedApplicationFailureRecord:
        record = self._build_failure_record(
            phase=phase,
            source_kind=source_kind,
            source_id=source_id,
            exc=exc,
            reason_code=reason_code,
            primary=False,
        )
        self.secondary_failures.append(record)
        if self.current_failure is not None and self.current_failure.primary:
            updated = self.current_failure.model_copy(
                update={"secondary_failure_count": len(self.secondary_failures)},
            )
            self.current_failure = updated
            self.last_failure = updated
        else:
            self.last_failure = record
        return record

    def clear_current_failure(self) -> None:
        self.current_failure = None

    def clear_primary_exception(self) -> None:
        self._primary_exception = None

    def reset_attempt_local_state(self) -> None:
        self.runtime_created = False
        self.runtime_started = False
        self.started_component_ids.clear()
        self.context_closed = False
        self.instance_lease_acquired = False
        self.instance_lease_released = False
        self.observer_task_count = 0
        self.secondary_failures.clear()

    def snapshot(
        self,
        *,
        lifecycle: HostedApplicationLifecycleSnapshot,
        health: HostedApplicationHealthSnapshot,
    ) -> HostedApplicationDiagnosticSnapshot:
        return HostedApplicationDiagnosticSnapshot(
            application_id=self.application_id,
            instance_id=self.instance_id,
            profile_digest=self.profile_digest,
            definition_digest=self.definition_digest,
            lifecycle=lifecycle,
            health=health,
            runtime_created=self.runtime_created,
            runtime_started=self.runtime_started,
            started_component_ids=tuple(self.started_component_ids),
            component_start_order=self.component_start_order,
            hook_ids_by_point=dict(self.hook_ids_by_point),
            event_subscription_ids=self.event_subscription_ids,
            current_failure=self.current_failure,
            last_failure=self.last_failure,
            secondary_failures=tuple(self.secondary_failures),
            context_closed=self.context_closed,
            instance_lease_acquired=self.instance_lease_acquired,
            instance_lease_released=self.instance_lease_released,
            observer_task_count=self.observer_task_count,
            active_operation_phase=self.active_operation_phase,
            snapshot_timestamp=self.clock.now(),
        )

    def _validate_failure_id(self, failure_id: str) -> str:
        try:
            return validate_bounded_identifier(failure_id, field_name="failure_id")
        except ValueError as exc:
            raise HostedApplicationDiagnosticError(str(exc)) from exc

    def _validate_diagnostic_timestamp(self, occurred_at: datetime) -> datetime:
        if occurred_at.tzinfo is None:
            raise HostedApplicationDiagnosticError(
                "diagnostic clock produced naive timestamp"
            )
        return occurred_at

    def _bounded_identifier_or_diagnostic(self, value: str, *, field_name: str) -> str:
        try:
            return validate_bounded_identifier(value, field_name=field_name)
        except ValueError as exc:
            raise HostedApplicationDiagnosticError(str(exc)) from exc

    def _build_failure_record(
        self,
        *,
        phase: HostedApplicationFailurePhase,
        source_kind: str,
        source_id: str,
        exc: BaseException,
        reason_code: str,
        primary: bool,
    ) -> HostedApplicationFailureRecord:
        safe_reason = self._bounded_identifier_or_diagnostic(reason_code, field_name="reason_code")
        secondary_count = len(self.secondary_failures) if primary else 0
        if secondary_count < 0:
            raise HostedApplicationDiagnosticError("secondary_failure_count must be non-negative")
        occurred_at = self._validate_diagnostic_timestamp(self.clock.now())
        return HostedApplicationFailureRecord(
            failure_id=self.next_failure_id(),
            reason_code=safe_reason,
            phase=phase,
            source_kind=self._bounded_identifier_or_diagnostic(
                source_kind, field_name="source_kind"
            ),
            source_id=self._bounded_identifier_or_diagnostic(source_id, field_name="source_id"),
            exception_type=type(exc).__name__,
            occurred_at=occurred_at,
            primary=primary,
            secondary_failure_count=secondary_count,
        )
