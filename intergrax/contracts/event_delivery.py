# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observability event delivery contracts (W5-A).

Transport-only boundary: no Kafka/OTLP/SQLite coupling. Durable evidence remains
owned by RuntimeEvent persistence, checkpoint CAS, lineage CAS, and external-operation
terminal stores — not by this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

OBSERVABILITY_EXPORT_PAYLOAD_SCHEMA: Literal["observability_export_payload.v1"] = (
    "observability_export_payload.v1"
)


class EventPriority(StrEnum):
    """Delivery class for observability transport (not execution scheduling)."""

    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT"
    BEST_EFFORT = "BEST_EFFORT"


class CriticalEventKind(StrEnum):
    """Canonical critical kinds — never silently dropped at the delivery boundary."""

    DECISION_FINALIZED = "DECISION_FINALIZED"
    SECURITY_EVENT = "SECURITY_EVENT"
    EXTERNAL_OPERATION_TERMINAL = "EXTERNAL_OPERATION_TERMINAL"
    CHECKPOINT_COMMITTED = "CHECKPOINT_COMMITTED"
    RECOVERY_STATE_CHANGE = "RECOVERY_STATE_CHANGE"


class EventDeliveryDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    DROPPED = "DROPPED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


class EventDeliveryObligation(StrEnum):
    """What ``ACCEPTED`` must mean for a given ``publish()`` call."""

    ADMISSION = "ADMISSION"
    COMPLETION = "COMPLETION"


class EventSinkHealthState(StrEnum):
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"


class EventDeliveryLateFailureStage(StrEnum):
    DOWNSTREAM_PUBLISH = "DOWNSTREAM_PUBLISH"


class EventDeliveryBoundaryFailureKind(StrEnum):
    """Normalized delivery-boundary failure (not persistence or OTLP vendor errors)."""

    SINK_CLOSED = "SINK_CLOSED"
    SINK_UNAVAILABLE = "SINK_UNAVAILABLE"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    COMPLETION_TIMEOUT = "COMPLETION_TIMEOUT"


class EventDeliveryReaction(StrEnum):
    CONTINUE = "CONTINUE"
    FAIL_EXECUTION = "FAIL_EXECUTION"


@dataclass(frozen=True, slots=True)
class ObservabilityExportSafeAttribute:
    key: str
    value: str | int


@dataclass(frozen=True, slots=True)
class ObservabilityExportPayload:
    """Vendor-neutral, redacted observability export payload (contract layer)."""

    event_id: str
    kind: str
    schema_version: Literal["observability_export_payload.v1"] = (
        OBSERVABILITY_EXPORT_PAYLOAD_SCHEMA
    )
    event_type: str = ""
    run_id: str = ""
    task_id: str = ""
    attempt_id: str = ""
    execution_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    correlation_id: str = ""
    parent_event_id: str = ""
    execution_phase: str = ""
    w3c_traceparent: str = ""
    w3c_tracestate: str = ""
    safe_attributes: tuple[ObservabilityExportSafeAttribute, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if not self.kind.strip():
            raise ValueError("kind must be non-empty")


@dataclass(frozen=True, slots=True)
class DeliverableEvent:
    """Opaque envelope for sink transport (distinct from ``RuntimeEvent`` evidence)."""

    export_payload: ObservabilityExportPayload
    sequence: int = 0

    @property
    def event_id(self) -> str:
        return self.export_payload.event_id

    @property
    def kind(self) -> str:
        return self.export_payload.kind


def make_observability_export_payload(
    *,
    event_id: str,
    kind: str,
    event_type: str = "",
    run_id: str = "",
    task_id: str = "",
    attempt_id: str = "",
    execution_id: str = "",
    agent_id: str = "",
    tenant_id: str = "",
    correlation_id: str = "",
    parent_event_id: str = "",
    execution_phase: str = "",
    w3c_traceparent: str = "",
    w3c_tracestate: str = "",
    safe_attributes: tuple[ObservabilityExportSafeAttribute, ...] = (),
) -> ObservabilityExportPayload:
    return ObservabilityExportPayload(
        event_id=event_id,
        kind=kind,
        event_type=event_type,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        parent_event_id=parent_event_id,
        execution_phase=execution_phase,
        w3c_traceparent=w3c_traceparent,
        w3c_tracestate=w3c_tracestate,
        safe_attributes=safe_attributes,
    )


def make_deliverable_event(
    export_payload: ObservabilityExportPayload,
    *,
    sequence: int = 0,
) -> DeliverableEvent:
    return DeliverableEvent(export_payload=export_payload, sequence=sequence)


@dataclass(frozen=True, slots=True)
class EventDeliveryResult:
    disposition: EventDeliveryDisposition
    priority: EventPriority
    buffered_depth: int
    obligation: EventDeliveryObligation


@dataclass(frozen=True, slots=True)
class EventDeliveryPolicy:
    """Bounded buffer + per-priority overflow semantics."""

    max_capacity: int
    important_wait_timeout_seconds: float = 0.05
    critical_completion_timeout_seconds: float = 5.0
    drain_shutdown_timeout_seconds: float = 30.0
    critical_reserved_capacity: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.max_capacity, bool):
            raise TypeError("max_capacity must be int, not bool")
        if isinstance(self.critical_reserved_capacity, bool):
            raise TypeError("critical_reserved_capacity must be int, not bool")
        if self.max_capacity < 1:
            raise ValueError("max_capacity must be >= 1")
        if self.critical_reserved_capacity < 0:
            raise ValueError("critical_reserved_capacity must be >= 0")
        if self.critical_reserved_capacity > self.max_capacity:
            raise ValueError("critical_reserved_capacity must be <= max_capacity")
        if self.important_wait_timeout_seconds < 0:
            raise ValueError("important_wait_timeout_seconds must be >= 0")
        if self.critical_completion_timeout_seconds <= 0:
            raise ValueError("critical_completion_timeout_seconds must be > 0")
        if self.drain_shutdown_timeout_seconds <= 0:
            raise ValueError("drain_shutdown_timeout_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class EventDeliveryLateFailure:
    deliverable: DeliverableEvent
    priority: EventPriority
    disposition: EventDeliveryDisposition
    stage: EventDeliveryLateFailureStage
    boundary_kind: EventDeliveryBoundaryFailureKind | None = None


class CriticalEventDeliveryError(RuntimeError):
    """Fail-closed when a critical event cannot be accepted (raised only by ``RuntimeEventBus``)."""


class EventDeliveryBoundaryError(Exception):
    """Normalized observability delivery failure at an ``EventSinkPort`` boundary."""

    kind: EventDeliveryBoundaryFailureKind
    message: str
    deliverable_event_id: str

    def __init__(
        self,
        *,
        kind: EventDeliveryBoundaryFailureKind,
        message: str,
        deliverable_event_id: str = "",
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.deliverable_event_id = deliverable_event_id


@runtime_checkable
class EventSinkPort(Protocol):
    """Pluggable observability event sink (W5-A)."""

    def publish(
        self,
        event: DeliverableEvent,
        *,
        priority: EventPriority,
        deadline: float | None = None,
    ) -> EventDeliveryResult: ...

    def close(self) -> None: ...


@runtime_checkable
class EventExportSinkPort(Protocol):
    """Downstream export transport (W5-C): delivery, flush, shutdown only."""

    async def export(self, payload: ObservabilityExportPayload) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class EventDeliveryObligationPolicyPort(Protocol):
    def obligation_for(self, priority: EventPriority) -> EventDeliveryObligation: ...


@runtime_checkable
class EventDeliveryAdmissionPolicyPort(Protocol):
    """Capacity partitioning for non-CRITICAL admission (CRITICAL may use reserved slots)."""

    def max_non_critical_buffered_events(self, policy: EventDeliveryPolicy) -> int: ...


@runtime_checkable
class EventDeliveryPostAdmissionFailureObserverPort(Protocol):
    def on_late_failure(self, failure: EventDeliveryLateFailure) -> None: ...


@runtime_checkable
class EventSinkHealthPort(Protocol):
    def health_state(self) -> EventSinkHealthState: ...

    def mark_unhealthy(self) -> None: ...


@runtime_checkable
class EventSinkDeliveryReactionPort(Protocol):
    """Extension point: interpret sink results without raising ``CriticalEventDeliveryError``."""

    def react_to_result(
        self,
        *,
        priority: EventPriority,
        result: EventDeliveryResult,
        deliverable: DeliverableEvent,
    ) -> EventDeliveryReaction: ...

    def react_to_boundary_error(
        self,
        *,
        priority: EventPriority,
        error: EventDeliveryBoundaryError,
        deliverable: DeliverableEvent,
    ) -> EventDeliveryReaction: ...


def effective_event_delivery_obligation(
    priority: EventPriority,
    policy: EventDeliveryObligationPolicyPort,
) -> EventDeliveryObligation:
    """Platform floor: ``CRITICAL`` always requires ``COMPLETION``."""
    obligation = policy.obligation_for(priority)
    if priority is EventPriority.CRITICAL:
        return EventDeliveryObligation.COMPLETION
    return obligation


def priority_for_critical_kind(kind: CriticalEventKind) -> EventPriority:
    return EventPriority.CRITICAL


def classify_kind_string(kind: str) -> EventPriority:
    """Map a kind label to delivery priority (critical kinds are frozen)."""
    try:
        CriticalEventKind(kind)
    except ValueError:
        pass
    else:
        return EventPriority.CRITICAL
    if kind.startswith("metrics.") or kind.startswith("timing."):
        return EventPriority.IMPORTANT
    return EventPriority.BEST_EFFORT
