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
    TaskRuntimeEventRuns,
    _filter_positioned_run_rows,
    _group_positioned_task_rows,
    _validate_persistence_tenant_id,
    _validate_run_list_params,
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
        after: ExecutionEventPosition | None = None,
    ) -> List[PositionedRuntimeEvent]:
        limit, through, after = _validate_run_list_params(
            limit=limit,
            through=through,
            after=after,
        )
        rows = self._by_run.get((tenant_id, run_id), [])
        return _filter_positioned_run_rows(
            rows,
            after=after,
            through=through,
            limit=limit,
        )

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        grouped = self.list_positioned_for_task_grouped_by_run(
            task_id,
            tenant_id=tenant_id,
            limit=limit,
        )
        events: list[RuntimeEvent] = []
        for _, run_rows in grouped.runs:
            for positioned in run_rows:
                events.append(positioned.event)
        return events

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        rows = self._by_task.get((tenant_id, task_id), [])
        return _group_positioned_task_rows(rows, limit=limit)

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
