# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime event package (architecture §42.1–§42.2)."""

from intergrax.runtime.events.event_catalog import (
    EVENT_CATALOG,
    EventCatalogEntry,
    EventCategory,
    RetentionClass,
    category_for_event_kind,
    category_for_spine_type,
    get_catalog_entry,
    ops_filter_hint_for_event,
    phase_for_event,
    should_persist_event,
)
from intergrax.runtime.events.event_kind_registry import (
    EventKindRegistryEntry,
    register_event_kind,
    require_registered_event_kind,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
    as_of_boundary_for_positioned,
)
from intergrax.runtime.events.persistence_contract import (
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
)
from intergrax.runtime.events.store import (
    DEFAULT_RUNTIME_EVENTS_DB,
    ENV_RUNTIME_EVENTS_DB,
    open_runtime_event_store,
    resolve_runtime_event_persistence,
    resolve_runtime_events_db_path,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.payload_registry import (
    EVENT_TYPE_PREFERRED_SCHEMA,
    get_payload_schema,
    register_payload_schema,
    runtime_event_with_payload,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.signals import emit_domain_signal, emit_platform_event
from intergrax.runtime.events.journal_query import query_journal
from intergrax.runtime.events.trace_bridge import (
    runtime_event_from_task_state,
    trace_event_to_runtime_event,
)
from intergrax.runtime.events.asof_projection import (
    AttemptAsOfSummary,
    HistoricalEventReference,
    InvalidRunExecutionHistoryError,
    RunExecutionAsOfProjection,
    RunExecutionHistoryNotFoundError,
    RunExecutionHistoryTruncatedError,
    RunExecutionLifecycleStatus,
    RunExecutionProjectionError,
    RunLifecycleViolationKind,
    apply_lifecycle_event,
    is_final_run_lifecycle_status,
    project_run_execution_as_of,
    reconstruct_run_execution_as_of,
)

from intergrax.runtime.events.unified_run_journal import (
    JOURNAL_SCHEMA_VERSION,
    PositionedJournalPrefixTruncatedError,
    build_unified_run_journal,
    load_positioned_run_journal_through,
)

__all__ = [
    "AsOfBoundary",
    "AttemptAsOfSummary",
    "DEFAULT_RUNTIME_EVENTS_DB",
    "ENV_RUNTIME_EVENTS_DB",
    "EVENT_CATALOG",
    "EVENT_TYPE_PREFERRED_SCHEMA",
    "EmitContext",
    "EventCatalogEntry",
    "EventCategory",
    "EventKindRegistryEntry",
    "ExecutionEventPosition",
    "ExecutionPhase",
    "HistoricalEventReference",
    "InMemoryRuntimeEventStore",
    "InvalidRunExecutionHistoryError",
    "JOURNAL_SCHEMA_VERSION",
    "NullRuntimeEventPersistence",
    "PositionedJournalPrefixTruncatedError",
    "PositionedRuntimeEvent",
    "RetentionClass",
    "RunExecutionAsOfProjection",
    "RunExecutionHistoryNotFoundError",
    "RunExecutionHistoryTruncatedError",
    "RunExecutionLifecycleStatus",
    "RunExecutionProjectionError",
    "RunLifecycleViolationKind",
    "apply_lifecycle_event",
    "is_final_run_lifecycle_status",
    "RuntimeEvent",
    "RuntimeEventBus",
    "RuntimeEventPayload",
    "RuntimeEventPersistence",
    "RuntimeEventType",
    "category_for_event_kind",
    "category_for_spine_type",
    "emit_domain_signal",
    "emit_platform_event",
    "get_catalog_entry",
    "get_payload_schema",
    "load_positioned_run_journal_through",
    "ops_filter_hint_for_event",
    "phase_for_event",
    "project_run_execution_as_of",
    "reconstruct_run_execution_as_of",
    "register_event_kind",
    "register_payload_schema",
    "require_registered_event_kind",
    "runtime_event_with_payload",
    "should_persist_event",
    "validate_payload_envelope",
    "SQLiteRuntimeEventStore",
    "as_of_boundary_for_positioned",
    "build_unified_run_journal",
    "open_runtime_event_store",
    "query_journal",
    "resolve_runtime_event_persistence",
    "resolve_runtime_events_db_path",
    "runtime_event_from_task_state",
    "trace_event_to_runtime_event",
]
