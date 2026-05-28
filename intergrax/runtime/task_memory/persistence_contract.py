# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-1 persistence contract for Nexus-owned task memory (§27, Phase I.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.runtime.task_memory.models import TaskMemoryRecord


class TaskMemoryPersistence(ABC):
    """
    Key/value store scoped by ``tenant_id`` + ``task_id`` + ``namespace``.

    Implementations: in-memory (tests), SQLite (lab). Nexus and future
    ``MemoryView`` (I.2) depend on this contract — not on a vendor backend.
    """

    @abstractmethod
    def put(self, record: TaskMemoryRecord) -> TaskMemoryRecord:
        """Insert or replace a record (unique on tenant/task/namespace/key)."""

    @abstractmethod
    def get(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> Optional[TaskMemoryRecord]:
        ...

    @abstractmethod
    def delete(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        ...

    @abstractmethod
    def list_records(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> List[TaskMemoryRecord]:
        """Return records in a namespace, optionally filtered by key prefix."""

    @abstractmethod
    def count_for_task(self, *, tenant_id: str, task_id: str) -> int:
        ...

    @abstractmethod
    def clear_task(self, *, tenant_id: str, task_id: str) -> int:
        """Remove all records for a task. Returns deleted count."""

    def close(self) -> None:
        """Release backend resources (no-op for most stores."""


class NullTaskMemoryPersistence(TaskMemoryPersistence):
    """Explicit no-op backend."""

    def put(self, record: TaskMemoryRecord) -> TaskMemoryRecord:
        _ = record
        return record

    def get(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> Optional[TaskMemoryRecord]:
        _ = tenant_id, task_id, namespace, key
        return None

    def delete(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> bool:
        _ = tenant_id, task_id, namespace, key
        return False

    def list_records(
        self,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> List[TaskMemoryRecord]:
        _ = tenant_id, task_id, namespace, prefix, limit
        return []

    def count_for_task(self, *, tenant_id: str, task_id: str) -> int:
        _ = tenant_id, task_id
        return 0

    def clear_task(self, *, tenant_id: str, task_id: str) -> int:
        _ = tenant_id, task_id
        return 0
