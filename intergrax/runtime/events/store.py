# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RuntimeEvent store opener and persistence resolution (§42.24)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore

ENV_RUNTIME_EVENTS_DB = "INTERGRAX_RUNTIME_EVENTS_DB"
DEFAULT_RUNTIME_EVENTS_DB = Path("build") / "intergrax_runtime_events.db"


def resolve_runtime_events_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_RUNTIME_EVENTS_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_RUNTIME_EVENTS_DB


def open_runtime_event_store(db_path: Path | None = None) -> SQLiteRuntimeEventStore:
    """Open the default SQLite RuntimeEvent backend (lab / local persistence)."""
    path = db_path or resolve_runtime_events_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteRuntimeEventStore(db_path=path)


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
