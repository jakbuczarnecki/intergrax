# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 persistence contract for canonical RuntimeEvent streams (§42.24)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.runtime.events.runtime_event import RuntimeEvent


class RuntimeEventPersistence(ABC):
    """
    Append-only runtime event store.

    Implementations (SQLite, Cassandra, Kafka, …) live behind this contract.
    Nexus and debug surfaces depend on the interface, not a specific backend.
    """

    @abstractmethod
    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        """Persist a single runtime event (idempotent on ``event_id`` when supported)."""

    @abstractmethod
    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        """Return events for a run scoped by tenant (oldest first)."""

    @abstractmethod
    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        """Return events for a task scoped by tenant (oldest first)."""

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""


class NullRuntimeEventPersistence(RuntimeEventPersistence):
    """Explicit no-op backend for tests and disabled persistence."""

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        _ = event, tenant_id

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        _ = run_id, tenant_id, limit
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


def resolve_event_tenant_id(event: RuntimeEvent, explicit: Optional[str] = None) -> str:
    """Resolve tenant scope for persistence (explicit > event field > empty)."""
    if explicit:
        return explicit
    if event.tenant_id:
        return event.tenant_id
    return ""
