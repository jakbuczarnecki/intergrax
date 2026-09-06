# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 persistence contract for canonical RuntimeEvent streams (§42.24)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from threading import Lock
from typing import List, Optional

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
    validate_execution_event_position,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent


class RuntimeEventPersistenceIntegrityError(Exception):
    """Raised when runtime event storage or derived indexes are inconsistent."""


class RuntimeEventPersistence(ABC):
    """
    Append-only runtime event store.

    Implementations (SQLite, Cassandra, Kafka, …) live behind this contract.
    Nexus and debug surfaces depend on the interface, not a specific backend.
    """

    @abstractmethod
    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        """
        Persist a single runtime event and return its canonical execution position.

        Idempotent on ``event_id``: duplicate append returns the original position.
        """

    @abstractmethod
    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        """Return positioned events for a run (oldest execution position first)."""

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        """Return events for a run scoped by tenant (canonical execution order)."""
        return [
            positioned.event
            for positioned in self.list_positioned_for_run(
                run_id,
                tenant_id=tenant_id,
                limit=limit,
            )
        ]

    @abstractmethod
    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        """Return events for a task scoped by tenant (canonical execution order)."""

    @abstractmethod
    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        """Return the accepted positioned event for ``tenant_id`` + ``event_id``, or ``None``."""

    def list_positioned_through(
        self,
        boundary: AsOfBoundary,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[PositionedRuntimeEvent]:
        """Return the inclusive execution-history prefix identified by ``boundary``."""
        return self.list_positioned_for_run(
            boundary.run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=boundary.position,
        )

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""


class NullRuntimeEventPersistence(RuntimeEventPersistence):
    """Explicit no-op backend for tests and disabled persistence."""

    def __init__(self) -> None:
        self._next_position: dict[tuple[str, str], int] = defaultdict(lambda: 1)
        self._accepted: dict[str, PositionedRuntimeEvent] = {}
        self._lock = Lock()

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        scope = tenant_id or event.tenant_id or ""
        with self._lock:
            existing = self._accepted.get(event.event_id)
            if existing is not None:
                return existing
            key = (scope, event.run_id)
            position = ExecutionEventPosition(self._next_position[key])
            self._next_position[key] += 1
            positioned = PositionedRuntimeEvent(event=event, position=position)
            self._accepted[event.event_id] = positioned
            return positioned

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        _ = run_id, tenant_id, limit, through
        return []

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        _ = task_id, tenant_id, limit
        return []

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        validated_tenant_id = _validate_persistence_tenant_id(tenant_id)
        validated_event_id = validate_event_id(event_id)
        positioned = self._accepted.get(str(validated_event_id))
        if positioned is None:
            return None
        if positioned.event.tenant_id != validated_tenant_id:
            return None
        return positioned


def _validate_persistence_tenant_id(tenant_id: object) -> str:
    if type(tenant_id) is not str:
        raise TypeError(f"tenant_id must be str, got {type(tenant_id).__name__}")
    if not tenant_id.strip():
        raise ValueError("tenant_id is required")
    if tenant_id != tenant_id.strip():
        raise ValueError("tenant_id must not contain leading or trailing whitespace")
    return tenant_id


def resolve_event_tenant_id(event: RuntimeEvent, explicit: Optional[str] = None) -> str:
    """Resolve tenant scope for persistence (explicit > event field > empty)."""
    if explicit:
        return explicit
    if event.tenant_id:
        return event.tenant_id
    return ""


def _validate_through_limit(
    *,
    limit: int,
    through: ExecutionEventPosition | None,
) -> tuple[int, ExecutionEventPosition | None]:
    if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be > 0")
    if through is None:
        return limit, None
    return limit, validate_execution_event_position(through)
