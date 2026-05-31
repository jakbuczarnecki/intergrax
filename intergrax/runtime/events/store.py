# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEvent store opener and persistence resolution (§42.24)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_runtime_event_store
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_RUNTIME_EVENTS_DB,
    ENV_RUNTIME_EVENTS_DB,
    resolve_runtime_events_db_path,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore

__all__ = [
    "DEFAULT_RUNTIME_EVENTS_DB",
    "ENV_RUNTIME_EVENTS_DB",
    "resolve_runtime_events_db_path",
    "open_runtime_event_store",
    "resolve_runtime_event_persistence",
]


def open_runtime_event_store(db_path: Path | None = None) -> SQLiteRuntimeEventStore:
    """Open SQLite runtime event store via ``integrations.providers.sqlite``."""
    if db_path is not None:
        return create_sqlite_runtime_event_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_runtime_event_store()  # type: ignore[return-value]


def resolve_runtime_event_persistence(
    *,
    db_path: Path | None = None,
    implementation: RuntimeEventPersistence | None = None,
) -> RuntimeEventPersistence | None:
    """
    Resolve RuntimeEvent persistence for composition roots (NexusLoop, debug API, tests).

    Priority: explicit ``implementation`` > SQLite at ``db_path``/env path.
    Pass ``implementation=InMemoryRuntimeEventStore()`` in unit tests.
    Pass ``implementation=None`` and omit path to disable persistence.
    """
    if implementation is not None:
        return implementation
    if db_path is None and not os.environ.get(ENV_RUNTIME_EVENTS_DB, "").strip():
        return None
    return open_runtime_event_store(db_path)
