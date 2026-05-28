# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TaskMemory store opener and SQLite backend (§27, Phase I.1)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import SQLiteTaskMemoryStore

ENV_TASK_MEMORY_DB = "INTERGRAX_TASK_MEMORY_DB"
DEFAULT_TASK_MEMORY_DB = Path("build") / "intergrax_task_memory.db"


def resolve_task_memory_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_TASK_MEMORY_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_TASK_MEMORY_DB


def open_task_memory_store(db_path: Path | None = None) -> SQLiteTaskMemoryStore:
    """Open the default SQLite TaskMemory backend (lab / local persistence)."""
    path = db_path or resolve_task_memory_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteTaskMemoryStore(db_path=path)


def resolve_task_memory_persistence(
    *,
    db_path: Path | None = None,
    implementation: TaskMemoryPersistence | None = None,
) -> TaskMemoryPersistence | None:
    """
    Resolve TaskMemory for composition roots (debug app, NexusLoop, tests).

    Priority: explicit ``implementation`` > SQLite at ``db_path``/env path.
    Pass ``implementation=InMemoryTaskMemoryStore()`` in unit tests.
    Pass ``implementation=None`` and omit path to disable persistence.
    """
    if implementation is not None:
        return implementation
    if db_path is None and not os.environ.get(ENV_TASK_MEMORY_DB, "").strip():
        return None
    return open_task_memory_store(db_path)
