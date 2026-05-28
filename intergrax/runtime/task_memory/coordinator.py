# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded read/write helpers for Nexus-owned task memory (§27)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from intergrax.runtime.task_memory.limits import (
    TaskMemoryLimits,
    validate_namespace_key,
    validate_value_size,
)
from intergrax.runtime.task_memory.models import TaskMemoryRecord, utc_now_iso
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence


class TaskMemoryCoordinator:
    """
    Enforces bounded writes before delegating to ``TaskMemoryPersistence``.

    ``PolicyScopedMemoryView`` (I.2) wraps this coordinator with policy + events.
    """

    @staticmethod
    def write(
        store: TaskMemoryPersistence,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
        value: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
        limits: Optional[TaskMemoryLimits] = None,
    ) -> TaskMemoryRecord:
        resolved_limits = limits or TaskMemoryLimits()
        validate_namespace_key(namespace=namespace, key=key, limits=resolved_limits)
        validate_value_size(value, limits=resolved_limits)

        existing = store.get(
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace.strip(),
            key=key.strip(),
        )
        if existing is None:
            count = store.count_for_task(tenant_id=tenant_id, task_id=task_id)
            if count >= resolved_limits.max_records_per_task:
                raise ValueError(
                    f"task memory record limit reached ({resolved_limits.max_records_per_task})"
                )

        now = utc_now_iso()
        record = TaskMemoryRecord(
            record_id=existing.record_id if existing is not None else f"tm_{uuid4().hex}",
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace.strip(),
            key=key.strip(),
            value=dict(value),
            created_at_utc=existing.created_at_utc if existing is not None else now,
            updated_at_utc=now,
            provenance=dict(provenance or {}),
        )
        return store.put(record)

    @staticmethod
    def read(
        store: TaskMemoryPersistence,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        key: str,
    ) -> Optional[TaskMemoryRecord]:
        return store.get(
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace.strip(),
            key=key.strip(),
        )

    @staticmethod
    def list_namespace(
        store: TaskMemoryPersistence,
        *,
        tenant_id: str,
        task_id: str,
        namespace: str,
        prefix: str = "",
        limit: int = 100,
    ) -> List[TaskMemoryRecord]:
        return store.list_records(
            tenant_id=tenant_id,
            task_id=task_id,
            namespace=namespace.strip(),
            prefix=prefix,
            limit=limit,
        )
