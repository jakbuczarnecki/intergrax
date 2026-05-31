# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Validating wrapper for RuntimeEventPersistence (§42.29)."""

from __future__ import annotations

from typing import List

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.schema_guard import assert_runtime_event_schema


class ValidatingRuntimeEventPersistence(RuntimeEventPersistence):
    """Enforces schema_version and ExecutionPhase mapping before append."""

    def __init__(self, inner: RuntimeEventPersistence) -> None:
        self._inner = inner

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        assert_runtime_event_schema(event)
        self._inner.append(event, tenant_id=tenant_id)

    def list_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return self._inner.list_for_run(run_id, tenant_id=tenant_id, limit=limit)

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> List[RuntimeEvent]:
        return self._inner.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def close(self) -> None:
        self._inner.close()
