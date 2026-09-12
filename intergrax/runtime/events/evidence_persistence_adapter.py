# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Adapter from ``RuntimeEventPersistence`` to ``EvidencePersistencePort``."""

from __future__ import annotations

from collections.abc import Callable
from typing import List, TypeVar

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    EvidencePersistenceIntegrityError,
)
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
    TaskRuntimeEventRuns,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent

__all__ = [
    "RuntimeEventPersistenceEvidenceAdapter",
    "as_evidence_persistence_port",
]

_T = TypeVar("_T")


def _invoke_with_boundary_isolation(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except EvidencePersistenceBoundaryError:
        raise
    except RuntimeEventPersistenceIntegrityError as exc:
        raise EvidencePersistenceIntegrityError(str(exc)) from exc
    except Exception as exc:
        raise EvidencePersistenceBoundaryError(
            "execution evidence persistence failed",
        ) from exc


class RuntimeEventPersistenceEvidenceAdapter:
    """
    Delegates execution evidence operations to an existing ``RuntimeEventPersistence``.

    Single adapter path: no parallel persistence implementations. Storage-layer
    exceptions are translated to ``EvidencePersistenceBoundaryError`` subclasses.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: RuntimeEventPersistence) -> None:
        self._inner = inner

    @property
    def inner(self) -> RuntimeEventPersistence:
        """Underlying store (composition roots, diagnostics wiring)."""
        return self._inner

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.append(event, tenant_id=tenant_id),
        )

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
        after: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.list_positioned_for_run(
                run_id,
                tenant_id=tenant_id,
                limit=limit,
                through=through,
                after=after,
            ),
        )

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.list_for_task(
                task_id,
                tenant_id=tenant_id,
                limit=limit,
            ),
        )

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.list_positioned_for_task_grouped_by_run(
                task_id,
                tenant_id=tenant_id,
                limit=limit,
            ),
        )

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.get_by_event_id(tenant_id=tenant_id, event_id=event_id),
        )

    def list_positioned_through(
        self,
        boundary: AsOfBoundary,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[PositionedRuntimeEvent]:
        return _invoke_with_boundary_isolation(
            lambda: self._inner.list_positioned_through(
                boundary,
                tenant_id=tenant_id,
                limit=limit,
            ),
        )

    def close(self) -> None:
        _invoke_with_boundary_isolation(self._inner.close)


def as_evidence_persistence_port(
    persistence: EvidencePersistencePort | RuntimeEventPersistence | None,
) -> EvidencePersistencePort | None:
    """
    Normalize composition input to the execution evidence port.

    Accepts legacy ``RuntimeEventPersistence`` at wiring boundaries so existing
    tests and hosts keep working without duplicating persistence flows.
    """
    if persistence is None:
        return None
    if isinstance(persistence, RuntimeEventPersistenceEvidenceAdapter):
        return persistence
    if isinstance(persistence, RuntimeEventPersistence):
        return RuntimeEventPersistenceEvidenceAdapter(persistence)
    return persistence
