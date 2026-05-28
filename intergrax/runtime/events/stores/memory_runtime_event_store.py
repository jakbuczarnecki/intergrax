# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory RuntimeEvent persistence (tests, local lab)."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent


class InMemoryRuntimeEventStore(RuntimeEventPersistence):
    def __init__(self) -> None:
        self._by_run: DefaultDict[tuple[str, str], List[RuntimeEvent]] = defaultdict(list)
        self._by_task: DefaultDict[tuple[str, str], List[RuntimeEvent]] = defaultdict(list)
        self._seen_event_ids: set[str] = set()

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        scope = tenant_id or event.tenant_id or ""
        if event.event_id in self._seen_event_ids:
            return
        self._seen_event_ids.add(event.event_id)
        self._by_run[(scope, event.run_id)].append(event)
        self._by_task[(scope, event.task_id)].append(event)

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        rows = self._by_run.get((tenant_id, run_id), [])
        return list(rows[:limit])

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        rows = self._by_task.get((tenant_id, task_id), [])
        return list(rows[:limit])
