# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical prepare + validate boundary for ``EvidencePersistencePort`` writes."""

from __future__ import annotations

from typing import List

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_identity import EventId
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.schema_guard import assert_runtime_event_schema
from intergrax.runtime.events.spine_payload_codec import prepare_canonical_production_write_event

__all__ = [
    "CanonicalRuntimeEventWriteValidatedPort",
    "ValidatingEvidencePersistencePort",
]


class CanonicalRuntimeEventWriteValidatedPort:
    """
    Marker for ports that enforce canonical production write semantics on ``append``.

    Used at composition boundaries to avoid duplicate validation wrappers.
    """


class ValidatingEvidencePersistencePort(CanonicalRuntimeEventWriteValidatedPort):
    """
    Prepare + strict schema validation before delegating durable writes.

    Storage providers implement persistence only; semantic integrity is platform-owned.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: EvidencePersistencePort) -> None:
        self._inner = inner

    @property
    def inner(self) -> EvidencePersistencePort:
        return self._inner

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        prepared = prepare_canonical_production_write_event(event)
        assert_runtime_event_schema(prepared)
        return self._inner.append(prepared, tenant_id=tenant_id)

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
        after: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        return self._inner.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
            after=after,
        )

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return self._inner.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return self._inner.list_positioned_for_task_grouped_by_run(
            task_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        return self._inner.get_by_event_id(tenant_id=tenant_id, event_id=event_id)

    def list_positioned_through(
        self,
        boundary: AsOfBoundary,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[PositionedRuntimeEvent]:
        return self._inner.list_positioned_through(
            boundary,
            tenant_id=tenant_id,
            limit=limit,
        )
