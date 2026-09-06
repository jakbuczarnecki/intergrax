# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory RuntimeEvent persistence (tests, local lab)."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict, List

from intergrax.contracts.execution_identity import EventId, validate_event_id
from intergrax.runtime.events.execution_position import (
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    AcceptedRuntimeEvent,
    RuntimeEventPersistence,
    _validate_persistence_tenant_id,
    _validate_through_limit,
    reconcile_idempotent_event_acceptance,
    resolve_persistence_scope,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent


class InMemoryRuntimeEventStore(RuntimeEventPersistence):
    def __init__(self) -> None:
        self._by_run: DefaultDict[tuple[str, str], list[PositionedRuntimeEvent]] = defaultdict(
            list
        )
        self._by_task: DefaultDict[tuple[str, str], list[PositionedRuntimeEvent]] = defaultdict(
            list
        )
        self._accepted_by_event_id: dict[str, AcceptedRuntimeEvent] = {}
        self._next_position: DefaultDict[tuple[str, str], int] = defaultdict(lambda: 1)
        self._run_locks: DefaultDict[tuple[str, str], Lock] = defaultdict(Lock)

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        scope = resolve_persistence_scope(event=event, tenant_id=tenant_id)
        run_key = (scope, event.run_id)
        with self._run_locks[run_key]:
            existing = self._accepted_by_event_id.get(event.event_id)
            if existing is not None:
                positioned = reconcile_idempotent_event_acceptance(
                    existing,
                    event,
                    persistence_tenant_id=scope,
                )
                self._repair_derived_indexes(
                    accepted=existing,
                    event=event,
                    persistence_tenant_id=scope,
                )
                return positioned
            position = ExecutionEventPosition(self._next_position[run_key])
            self._next_position[run_key] += 1
            positioned = PositionedRuntimeEvent(event=event, position=position)
            accepted = AcceptedRuntimeEvent(tenant_id=scope, positioned=positioned)
            self._accepted_by_event_id[event.event_id] = accepted
            self._by_run[run_key].append(positioned)
            self._by_task[(scope, event.task_id)].append(positioned)
            return positioned

    def _repair_derived_indexes(
        self,
        *,
        accepted: AcceptedRuntimeEvent,
        event: RuntimeEvent,
        persistence_tenant_id: str,
    ) -> None:
        positioned = accepted.positioned
        run_key = (persistence_tenant_id, event.run_id)
        if positioned not in self._by_run[run_key]:
            self._by_run[run_key].append(positioned)
        task_key = (persistence_tenant_id, event.task_id)
        if positioned not in self._by_task[task_key]:
            self._by_task[task_key].append(positioned)

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        limit, through = _validate_through_limit(limit=limit, through=through)
        rows = self._by_run.get((tenant_id, run_id), [])
        if through is None:
            return list(rows[:limit])
        filtered = [row for row in rows if row.position <= through]
        return list(filtered[:limit])

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        if type(limit) is not int or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be > 0")
        rows = self._by_task.get((tenant_id, task_id), [])
        return [positioned.event for positioned in rows[:limit]]

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        validated_tenant_id = _validate_persistence_tenant_id(tenant_id)
        validated_event_id = validate_event_id(event_id)
        accepted = self._accepted_by_event_id.get(str(validated_event_id))
        if accepted is None:
            return None
        if accepted.tenant_id != validated_tenant_id:
            return None
        return accepted.positioned
