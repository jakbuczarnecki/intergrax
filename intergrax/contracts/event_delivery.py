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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.runtime.events.runtime_event import RuntimeEvent


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


@dataclass(frozen=True, slots=True)
class DeliverableEvent:
    """Opaque envelope for sink transport (distinct from ``RuntimeEvent`` evidence)."""

    event_id: str
    kind: str
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class EventDeliveryResult:
    disposition: EventDeliveryDisposition
    priority: EventPriority
    buffered_depth: int


@dataclass(frozen=True, slots=True)
class EventDeliveryPolicy:
    """Bounded buffer + per-priority overflow semantics."""

    max_capacity: int
    important_wait_timeout_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.max_capacity < 1:
            raise ValueError("max_capacity must be >= 1")
        if self.important_wait_timeout_seconds < 0:
            raise ValueError("important_wait_timeout_seconds must be >= 0")


class CriticalEventDeliveryError(RuntimeError):
    """Fail-closed when a critical event cannot be accepted (buffer saturated)."""


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

    async def export(self, event: RuntimeEvent) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


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
