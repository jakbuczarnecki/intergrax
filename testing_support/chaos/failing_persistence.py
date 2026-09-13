# © Artur Czarnecki. All rights reserved.

"""RuntimeEventPersistence fault wrapper (deterministic append failures)."""

from __future__ import annotations

from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistence,
    TaskRuntimeEventRuns,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from testing_support.chaos.fault_plan import FailOnCall, call_counter, raise_on_call


class FailOnAppendPersistence(RuntimeEventPersistence):
    """Delegate reads; fail append on configured call index."""

    def __init__(
        self,
        inner: RuntimeEventPersistence,
        *,
        fail_on: FailOnCall,
    ) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self._counter = call_counter()

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> None:
        raise_on_call(self._fail_on, counter=self._counter)
        self._inner.append(event, tenant_id=tenant_id)

    def list_positioned_for_run(
        self,
        run_id,
        *,
        tenant_id: str,
        limit: int = 1000,
        through=None,
        after=None,
    ):
        return self._inner.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
            after=after,
        )

    def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
        return self._inner.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return self._inner.list_positioned_for_task_grouped_by_run(
            task_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return self._inner.get_by_event_id(tenant_id=tenant_id, event_id=event_id)
