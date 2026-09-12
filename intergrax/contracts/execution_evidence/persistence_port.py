# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution evidence persistence port — enterprise storage boundary (NPSC-5F)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import EventId

if TYPE_CHECKING:
    from intergrax.contracts.execution_event_position import (
        AsOfBoundary,
        ExecutionEventPosition,
    )
    from intergrax.runtime.events.execution_position import PositionedRuntimeEvent
    from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns
    from intergrax.runtime.events.runtime_event import RuntimeEvent

__all__ = ["EvidencePersistencePort"]


@runtime_checkable
class EvidencePersistencePort(Protocol):
    """
    Stable persistence contract for canonical execution evidence (``RuntimeEvent``).

    Execution producers and the event bus depend on this port, not on a concrete
    storage backend. Adapters delegate to ``RuntimeEventPersistence`` implementations
    and translate storage failures to ``EvidencePersistenceBoundaryError`` subclasses.
    """

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        """Persist one runtime event; idempotent on ``event_id``.

        Raises ``EvidencePersistenceBoundaryError`` subclasses on persistence failure.
        """

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
        after: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        """Return positioned events for a run (oldest execution position first)."""

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        """Return events for a task scoped by tenant."""

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        """Return task events grouped by run with run-local execution order."""

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
