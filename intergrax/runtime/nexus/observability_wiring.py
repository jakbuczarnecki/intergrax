# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Nexus observability wiring for Tier-3 application factories (Appendix B.08, B.10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.store import (
    open_runtime_event_store,
    resolve_runtime_event_persistence,
    resolve_runtime_events_db_path,
)
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.nexus.tracing.store import open_run_trace_store, resolve_trace_db_path


@dataclass(frozen=True)
class NexusObservabilityStores:
    """Trace + runtime event backends wired into ``NexusLoop`` and debug API."""

    trace_store: RunTraceWriter
    runtime_event_store: RuntimeEventPersistence | None
    trace_db_path: Path | None
    runtime_events_db_path: Path | None


def wire_nexus_observability(
    *,
    trace_db_path: Path | None = None,
    runtime_events_db_path: Path | None = None,
    trace_store: RunTraceWriter | None = None,
    runtime_event_store: RuntimeEventPersistence | None = None,
    use_in_memory_trace: bool = False,
    enable_runtime_events: bool = True,
) -> NexusObservabilityStores:
    """
    Resolve observability stores for application composition roots.

    Default (production / local dev): SQLite trace + SQLite runtime events under ``build/``
    or ``INTERGRAX_*_DB`` env paths. Tests may pass explicit ``*_db_path`` or opt into
    ``use_in_memory_trace=True`` for isolation.
    """
    if trace_store is not None:
        resolved_trace = trace_store
        resolved_trace_path = trace_db_path
    elif use_in_memory_trace:
        resolved_trace = InMemoryRunTraceStore()
        resolved_trace_path = None
    else:
        path = trace_db_path or resolve_trace_db_path(None)
        resolved_trace = open_run_trace_store(path)
        resolved_trace_path = path

    if runtime_event_store is not None:
        resolved_events = runtime_event_store
        resolved_events_path = runtime_events_db_path
    elif not enable_runtime_events:
        resolved_events = None
        resolved_events_path = None
    else:
        path = runtime_events_db_path or resolve_runtime_events_db_path(None)
        resolved_events = resolve_runtime_event_persistence(db_path=path)
        resolved_events_path = path

    return NexusObservabilityStores(
        trace_store=resolved_trace,
        runtime_event_store=resolved_events,
        trace_db_path=resolved_trace_path,
        runtime_events_db_path=resolved_events_path,
    )
