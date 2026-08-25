# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Logical execution-state projection at an inclusive AsOfBoundary (TRACE-ASOF-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.unified_run_journal import (
    PositionedJournalBoundaryNotFoundError,
    PositionedJournalPrefixTruncatedError,
    load_positioned_run_journal_through,
)


class RunExecutionProjectionError(Exception):
    """Base error for as-of execution projection."""


class RunExecutionHistoryNotFoundError(RunExecutionProjectionError):
    """No canonical positioned history exists for the requested run."""


class RunExecutionBoundaryNotFoundError(RunExecutionProjectionError):
    """Canonical history exists but has no accepted event at the requested boundary position."""


class RunExecutionHistoryTruncatedError(RunExecutionProjectionError):
    """Persistence read limit truncated the prefix required for reconstruction."""


class RunLifecycleViolationKind(StrEnum):
    """Typed lifecycle transition violations raised by ``apply_lifecycle_event``."""

    EVENT_AFTER_TERMINAL = "event_after_terminal"
    CONFLICTING_FINAL_OUTCOME = "conflicting_final_outcome"
    DISALLOWED_AFTER_FAILED = "disallowed_after_failed"


class InvalidRunExecutionHistoryError(RunExecutionProjectionError):
    """Positioned input violates canonical execution-history ordering or identity."""

    def __init__(
        self,
        message: str,
        *,
        kind: RunLifecycleViolationKind | None = None,
        current_status: RunExecutionLifecycleStatus | None = None,
        event_type: RuntimeEventType | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.current_status = current_status
        self.event_type = event_type


class RunExecutionLifecycleStatus(str, Enum):
    """Closed run-level lifecycle status derived from ``RuntimeEventType`` only."""

    CREATED = "created"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    CANCELLATION_REQUESTED = "cancellation_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_FINAL_RUN_STATUSES = frozenset(
    {
        RunExecutionLifecycleStatus.COMPLETED,
        RunExecutionLifecycleStatus.CANCELLED,
    }
)

_LIFECYCLE_STATUS_BY_EVENT: dict[RuntimeEventType, RunExecutionLifecycleStatus] = {
    RuntimeEventType.TASK_CREATED: RunExecutionLifecycleStatus.CREATED,
    RuntimeEventType.PAUSE_REQUESTED: RunExecutionLifecycleStatus.PAUSE_REQUESTED,
    RuntimeEventType.PAUSED: RunExecutionLifecycleStatus.PAUSED,
    RuntimeEventType.RESUMED: RunExecutionLifecycleStatus.RUNNING,
    RuntimeEventType.CANCELLATION_REQUESTED: RunExecutionLifecycleStatus.CANCELLATION_REQUESTED,
    RuntimeEventType.CANCELLED: RunExecutionLifecycleStatus.CANCELLED,
    RuntimeEventType.TASK_COMPLETED: RunExecutionLifecycleStatus.COMPLETED,
    RuntimeEventType.TASK_FAILED: RunExecutionLifecycleStatus.FAILED,
    RuntimeEventType.PLAN_FAILED: RunExecutionLifecycleStatus.FAILED,
    RuntimeEventType.RETRY_STARTED: RunExecutionLifecycleStatus.RUNNING,
}


@dataclass(frozen=True, slots=True)
class HistoricalEventReference:
    """Typed provenance reference to one canonical positioned source event."""

    event_id: EventId
    position: ExecutionEventPosition
    attempt_id: AttemptId
    event_type: RuntimeEventType


@dataclass(frozen=True, slots=True)
class AttemptAsOfSummary:
    """Immutable attempt facts reconstructed from the canonical prefix."""

    attempt_id: AttemptId
    first_position: ExecutionEventPosition
    last_position: ExecutionEventPosition
    event_count: int


@dataclass(frozen=True, slots=True)
class RunExecutionAsOfProjection:
    """Immutable run execution / lifecycle state as of an inclusive boundary."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    boundary: AsOfBoundary
    last_included_position: ExecutionEventPosition
    lifecycle_status: RunExecutionLifecycleStatus
    attempt_ids: tuple[AttemptId, ...]
    current_attempt_id: AttemptId
    attempts: tuple[AttemptAsOfSummary, ...]
    source_events: tuple[HistoricalEventReference, ...]

    @property
    def is_terminal(self) -> bool:
        return is_final_run_lifecycle_status(self.lifecycle_status)


def project_run_execution_as_of(
    *,
    boundary: AsOfBoundary,
    positioned_events: tuple[PositionedRuntimeEvent, ...],
    tenant_id: str,
) -> RunExecutionAsOfProjection:
    """
    Pure reducer: canonical positioned prefix + boundary -> immutable projection.

    ``positioned_events`` MUST already be filtered to ``position <= boundary.position``.
    """
    _require_tenant_id(tenant_id)
    if not positioned_events:
        raise RunExecutionHistoryNotFoundError(
            f"no canonical execution history for run {boundary.run_id!r} through "
            f"position {boundary.position.value}"
        )
    _validate_positioned_input(
        positioned_events=positioned_events,
        boundary=boundary,
        tenant_id=tenant_id,
    )
    if positioned_events[-1].position != boundary.position:
        raise RunExecutionBoundaryNotFoundError(
            f"positioned prefix ends at {positioned_events[-1].position.value}, "
            f"not at requested boundary {boundary.position.value} "
            f"for run {boundary.run_id!r}"
        )

    lifecycle_status = RunExecutionLifecycleStatus.CREATED
    attempt_order: list[AttemptId] = []
    attempt_seen: set[AttemptId] = set()
    attempt_stats: dict[AttemptId, tuple[ExecutionEventPosition, ExecutionEventPosition, int]] = {}
    source_events: list[HistoricalEventReference] = []
    current_attempt_id: AttemptId | None = None
    task_id: TaskId | None = None

    for positioned in positioned_events:
        event = positioned.event
        if task_id is None:
            task_id = event.task_id
        elif event.task_id != task_id:
            raise InvalidRunExecutionHistoryError("inconsistent task_id in positioned prefix")

        attempt_id = event.attempt_id
        if attempt_id not in attempt_seen:
            attempt_seen.add(attempt_id)
            attempt_order.append(attempt_id)
            attempt_stats[attempt_id] = (positioned.position, positioned.position, 1)
        else:
            first_pos, _last_pos, count = attempt_stats[attempt_id]
            attempt_stats[attempt_id] = (first_pos, positioned.position, count + 1)

        lifecycle_status = apply_lifecycle_event(
            lifecycle_status,
            event.event_type,
        )
        current_attempt_id = attempt_id
        source_events.append(
            HistoricalEventReference(
                event_id=event.event_id,
                position=positioned.position,
                attempt_id=attempt_id,
                event_type=event.event_type,
            )
        )

    assert task_id is not None
    assert current_attempt_id is not None
    attempts = tuple(
        AttemptAsOfSummary(
            attempt_id=attempt,
            first_position=attempt_stats[attempt][0],
            last_position=attempt_stats[attempt][1],
            event_count=attempt_stats[attempt][2],
        )
        for attempt in attempt_order
    )
    return RunExecutionAsOfProjection(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=boundary.run_id,
        boundary=boundary,
        last_included_position=positioned_events[-1].position,
        lifecycle_status=lifecycle_status,
        attempt_ids=tuple(attempt_order),
        current_attempt_id=current_attempt_id,
        attempts=attempts,
        source_events=tuple(source_events),
    )


def reconstruct_run_execution_as_of(
    *,
    persistence: RuntimeEventPersistence,
    tenant_id: str,
    boundary: AsOfBoundary,
    initial_limit: int = 1000,
    max_limit: int = 1_000_000,
) -> RunExecutionAsOfProjection:
    """Load a complete positioned prefix and reduce it to an execution projection."""
    try:
        positioned_events = load_positioned_run_journal_through(
            persistence,
            tenant_id=tenant_id,
            boundary=boundary,
            initial_limit=initial_limit,
            max_limit=max_limit,
        )
    except PositionedJournalPrefixTruncatedError as exc:
        raise RunExecutionHistoryTruncatedError(str(exc)) from exc
    except PositionedJournalBoundaryNotFoundError as exc:
        raise RunExecutionBoundaryNotFoundError(str(exc)) from exc
    if not positioned_events:
        raise RunExecutionHistoryNotFoundError(
            f"no canonical execution history for run {boundary.run_id!r}"
        )
    return project_run_execution_as_of(
        boundary=boundary,
        positioned_events=positioned_events,
        tenant_id=tenant_id,
    )


def is_final_run_lifecycle_status(status: RunExecutionLifecycleStatus) -> bool:
    """Whether ``status`` is a terminal run closure (``COMPLETED`` or ``CANCELLED``)."""
    return status in _FINAL_RUN_STATUSES


def apply_lifecycle_event(
    current: RunExecutionLifecycleStatus,
    event_type: RuntimeEventType,
) -> RunExecutionLifecycleStatus:
    """Apply one canonical lifecycle transition; raise typed error on violation."""
    if event_type == RuntimeEventType.RETRY_SCHEDULED:
        if current in _FINAL_RUN_STATUSES:
            raise InvalidRunExecutionHistoryError(
                f"retry lifecycle event {event_type.value!r} after terminal "
                f"{current.value}",
                kind=RunLifecycleViolationKind.EVENT_AFTER_TERMINAL,
                current_status=current,
                event_type=event_type,
            )
        return current

    mapped = _LIFECYCLE_STATUS_BY_EVENT.get(event_type)
    if mapped is not None:
        if current in _FINAL_RUN_STATUSES:
            raise InvalidRunExecutionHistoryError(
                f"lifecycle event {event_type.value!r} after terminal {current.value}",
                kind=RunLifecycleViolationKind.EVENT_AFTER_TERMINAL,
                current_status=current,
                event_type=event_type,
            )
        if current == RunExecutionLifecycleStatus.FAILED:
            if event_type == RuntimeEventType.RETRY_STARTED:
                return RunExecutionLifecycleStatus.RUNNING
            if mapped in _FINAL_RUN_STATUSES:
                raise InvalidRunExecutionHistoryError(
                    f"conflicting final lifecycle event {event_type.value!r} after "
                    f"{current.value}",
                    kind=RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME,
                    current_status=current,
                    event_type=event_type,
                )
            if mapped != RunExecutionLifecycleStatus.FAILED:
                raise InvalidRunExecutionHistoryError(
                    f"disallowed lifecycle event {event_type.value!r} after "
                    f"{current.value}",
                    kind=RunLifecycleViolationKind.DISALLOWED_AFTER_FAILED,
                    current_status=current,
                    event_type=event_type,
                )
            return mapped
        return mapped

    if current == RunExecutionLifecycleStatus.CREATED:
        return RunExecutionLifecycleStatus.RUNNING
    if current == RunExecutionLifecycleStatus.FAILED:
        raise InvalidRunExecutionHistoryError(
            f"disallowed lifecycle event {event_type.value!r} after {current.value}",
            kind=RunLifecycleViolationKind.DISALLOWED_AFTER_FAILED,
            current_status=current,
            event_type=event_type,
        )
    return current


def _validate_positioned_input(
    *,
    positioned_events: tuple[PositionedRuntimeEvent, ...],
    boundary: AsOfBoundary,
    tenant_id: str,
) -> None:
    previous_position: ExecutionEventPosition | None = None
    for positioned in positioned_events:
        if positioned.run_id != boundary.run_id:
            raise InvalidRunExecutionHistoryError(
                "positioned event run_id does not match AsOfBoundary run_id"
            )
        event_tenant = positioned.event.tenant_id
        if event_tenant is not None and event_tenant != tenant_id:
            raise InvalidRunExecutionHistoryError(
                "positioned event tenant_id does not match requested tenant scope"
            )
        if positioned.position > boundary.position:
            raise InvalidRunExecutionHistoryError(
                "positioned event exceeds inclusive AsOfBoundary"
            )
        if previous_position is not None:
            if positioned.position == previous_position:
                raise InvalidRunExecutionHistoryError("duplicate ExecutionEventPosition")
            if positioned.position < previous_position:
                raise InvalidRunExecutionHistoryError(
                    "positioned events are not in strict increasing ExecutionEventPosition order"
                )
        previous_position = positioned.position


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    return tenant_id
