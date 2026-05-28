# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory and configuration for RuntimeEvent persistence backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from intergrax.runtime.events.persistence_contract import (
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import (
    InMemoryRuntimeEventStore,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import (
    SQLiteRuntimeEventStore,
)

ENV_RUNTIME_EVENT_STORE = "INTERGRAX_RUNTIME_EVENT_STORE"
ENV_RUNTIME_EVENTS_DB = "INTERGRAX_RUNTIME_EVENTS_DB"
DEFAULT_RUNTIME_EVENTS_DB = Path("build") / "intergrax_runtime_events.db"


class RuntimeEventStoreBackend(str, Enum):
    NONE = "none"
    MEMORY = "memory"
    SQLITE = "sqlite"


RuntimeEventStoreFactory = Callable[[], RuntimeEventPersistence]


@dataclass(frozen=True)
class RuntimeEventStoreSettings:
    backend: RuntimeEventStoreBackend = RuntimeEventStoreBackend.NONE
    sqlite_path: Optional[Path] = None


def resolve_runtime_events_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_RUNTIME_EVENTS_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_RUNTIME_EVENTS_DB


def resolve_runtime_event_store_settings(
    *,
    backend: Optional[str] = None,
    sqlite_path: Path | None = None,
) -> RuntimeEventStoreSettings:
    raw_backend = (
        backend
        or os.environ.get(ENV_RUNTIME_EVENT_STORE, RuntimeEventStoreBackend.NONE.value)
    ).strip().lower()
    try:
        resolved_backend = RuntimeEventStoreBackend(raw_backend)
    except ValueError:
        resolved_backend = RuntimeEventStoreBackend.NONE
    return RuntimeEventStoreSettings(
        backend=resolved_backend,
        sqlite_path=sqlite_path or resolve_runtime_events_db_path(None),
    )


def create_runtime_event_store(
    settings: Optional[RuntimeEventStoreSettings] = None,
    *,
    implementation: Optional[RuntimeEventPersistence] = None,
    factory: Optional[RuntimeEventStoreFactory] = None,
) -> Optional[RuntimeEventPersistence]:
    """
    Build a persistence backend.

    Priority: explicit ``implementation`` > ``factory`` > ``settings``/env defaults.
    Returns ``None`` when backend is ``none`` (in-process history only).
    """
    if implementation is not None:
        return implementation
    if factory is not None:
        return factory()
    resolved = settings or resolve_runtime_event_store_settings()
    if resolved.backend == RuntimeEventStoreBackend.NONE:
        return None
    if resolved.backend == RuntimeEventStoreBackend.MEMORY:
        return InMemoryRuntimeEventStore()
    if resolved.backend == RuntimeEventStoreBackend.SQLITE:
        path = resolved.sqlite_path or resolve_runtime_events_db_path(None)
        path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteRuntimeEventStore(db_path=path)
    return None


def open_runtime_event_store(db_path: Path | None = None) -> SQLiteRuntimeEventStore:
    """Convenience opener for the SQLite lab backend."""
    path = db_path or resolve_runtime_events_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteRuntimeEventStore(db_path=path)
