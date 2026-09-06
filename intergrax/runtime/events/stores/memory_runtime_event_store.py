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
    RuntimeEventPersistence,
    _validate_persistence_tenant_id,
    _validate_through_limit,
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
        self._accepted_by_event_id: dict[str, PositionedRuntimeEvent] = {}
        self._next_position: DefaultDict[tuple[str, str], int] = defaultdict(lambda: 1)
        self._run_locks: DefaultDict[tuple[str, str], Lock] = defaultdict(Lock)

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        scope = tenant_id or event.tenant_id or ""
        run_key = (scope, event.run_id)
        with self._run_locks[run_key]:
            existing = self._accepted_by_event_id.get(event.event_id)
            if existing is not None:
                return existing
            position = ExecutionEventPosition(self._next_position[run_key])
            self._next_position[run_key] += 1
            positioned = PositionedRuntimeEvent(event=event, position=position)
            self._accepted_by_event_id[event.event_id] = positioned
            self._by_run[run_key].append(positioned)
            self._by_task[(scope, event.task_id)].append(positioned)
            return positioned

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
        positioned = self._accepted_by_event_id.get(str(validated_event_id))
        if positioned is None:
            return None
        if positioned.event.tenant_id != validated_tenant_id:
            return None
        return positioned
