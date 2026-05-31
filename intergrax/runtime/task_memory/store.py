# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TaskMemory store opener and SQLite backend (§27, Phase I.1)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_TASK_MEMORY_DB,
    ENV_TASK_MEMORY_DB,
    resolve_task_memory_db_path,
)
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import SQLiteTaskMemoryStore

__all__ = [
    "DEFAULT_TASK_MEMORY_DB",
    "ENV_TASK_MEMORY_DB",
    "resolve_task_memory_db_path",
    "open_task_memory_store",
    "resolve_task_memory_persistence",
]


def open_task_memory_store(db_path: Path | None = None) -> SQLiteTaskMemoryStore:
    """Open SQLite TaskMemory via ``integrations.providers.sqlite``."""
    from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_task_memory_store

    if db_path is not None:
        return create_sqlite_task_memory_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_task_memory_store()  # type: ignore[return-value]


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
