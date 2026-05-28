# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory TaskMemory persistence (tests, local lab)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence


class InMemoryTaskMemoryStore(TaskMemoryPersistence):
    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str, str, str], TaskMemoryRecord] = {}

    @staticmethod
    def _key(tenant_id: str, task_id: str, namespace: str, key: str) -> Tuple[str, str, str, str]:
        return tenant_id, task_id, namespace, key

    def put(self, record: TaskMemoryRecord) -> TaskMemoryRecord:
        slot = self._key(record.tenant_id, record.task_id, record.namespace, record.key)
        self._records[slot] = record
        return record

    def get(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> Optional[TaskMemoryRecord]:
        return self._records.get(self._key(tenant_id, task_id, namespace, key))

    def delete(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        slot = self._key(tenant_id, task_id, namespace, key)
        if slot not in self._records:
            return False
        del self._records[slot]
        return True

    def list_records(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> List[TaskMemoryRecord]:
        rows = [
            record
            for record in self._records.values()
            if record.tenant_id == tenant_id
            and record.task_id == task_id
            and record.namespace == namespace
            and record.key.startswith(prefix)
        ]
        rows.sort(key=lambda item: item.key)
        return rows[:limit]

    def count_for_task(self, *, tenant_id: str, task_id: str) -> int:
        return sum(
            1
            for record in self._records.values()
            if record.tenant_id == tenant_id and record.task_id == task_id
        )

    def clear_task(self, *, tenant_id: str, task_id: str) -> int:
        to_delete = [
            slot
            for slot, record in self._records.items()
            if record.tenant_id == tenant_id and record.task_id == task_id
        ]
        for slot in to_delete:
            del self._records[slot]
        return len(to_delete)
