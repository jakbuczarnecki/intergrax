# © Artur Czarnecki. All rights reserved.

"""Lightweight runtime-event reads for cross-process spine subprocess CLIs."""

from __future__ import annotations

from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore


def terminal_event_type_for_run(
    *,
    runtime_store: SQLiteRuntimeEventStore,
    tenant_id: str,
    run_id: str,
) -> RuntimeEventType | None:
    terminal = {
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.TASK_FAILED,
    }
    for event in runtime_store.list_for_run(run_id, tenant_id=tenant_id):
        if event.event_type in terminal:
            return event.event_type
    return None


__all__ = ["terminal_event_type_for_run"]
