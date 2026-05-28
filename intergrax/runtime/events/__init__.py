# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event package (architecture §42.1–§42.2)."""

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import (
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
)
from intergrax.runtime.events.store_factory import (
    DEFAULT_RUNTIME_EVENTS_DB,
    ENV_RUNTIME_EVENTS_DB,
    ENV_RUNTIME_EVENT_STORE,
    RuntimeEventStoreBackend,
    RuntimeEventStoreSettings,
    create_runtime_event_store,
    open_runtime_event_store,
    resolve_runtime_event_store_settings,
    resolve_runtime_events_db_path,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.trace_bridge import (
    runtime_event_from_task_state,
    trace_event_to_runtime_event,
)

__all__ = [
    "DEFAULT_RUNTIME_EVENTS_DB",
    "ENV_RUNTIME_EVENTS_DB",
    "ENV_RUNTIME_EVENT_STORE",
    "ExecutionPhase",
    "NullRuntimeEventPersistence",
    "RuntimeEvent",
    "RuntimeEventBus",
    "RuntimeEventPersistence",
    "RuntimeEventStoreBackend",
    "RuntimeEventStoreSettings",
    "RuntimeEventType",
    "create_runtime_event_store",
    "open_runtime_event_store",
    "resolve_runtime_event_store_settings",
    "resolve_runtime_events_db_path",
    "runtime_event_from_task_state",
    "trace_event_to_runtime_event",
]
