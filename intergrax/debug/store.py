# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve debug persistence backends from env, explicit paths, or injected adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.store_factory import (
    RuntimeEventStoreBackend,
    RuntimeEventStoreSettings,
    create_runtime_event_store,
    resolve_runtime_event_store_settings,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
    TaskCheckpointReader,
)
from intergrax.runtime.long_running.store import (
    open_task_checkpoint_store,
    resolve_task_checkpoints_db_path,
)
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceReader
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore

ENV_TRACE_DB = "INTERGRAX_TRACE_DB"
DEFAULT_TRACE_DB = Path("build") / "intergrax_trace.db"


def resolve_trace_db_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_TRACE_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_TRACE_DB


def open_trace_reader(db_path: Path | None = None) -> RunTraceReader:
    path = db_path or resolve_trace_db_path(None)
    if not path.exists():
        raise FileNotFoundError(
            f"Trace database not found: {path}. "
            f"Set {ENV_TRACE_DB} or pass --db. Runs must be finalized via NexusLoop + trace_store."
        )
    return SQLiteRunTraceStore(db_path=path)


def open_runtime_event_persistence(
    *,
    db_path: Path | None = None,
    implementation: RuntimeEventPersistence | None = None,
) -> RuntimeEventPersistence | None:
    """
    Resolve runtime event store for debug API.

    Priority: explicit implementation > explicit db_path (sqlite) > env defaults.
    """
    if implementation is not None:
        return implementation
    if db_path is not None:
        return create_runtime_event_store(
            RuntimeEventStoreSettings(
                backend=RuntimeEventStoreBackend.SQLITE,
                sqlite_path=db_path,
            )
        )
    return create_runtime_event_store(resolve_runtime_event_store_settings())


def open_task_checkpoint_persistence(
    *,
    db_path: Path | None = None,
    implementation: TaskCheckpointPersistence | None = None,
) -> TaskCheckpointPersistence | None:
    if implementation is not None:
        return implementation
    if db_path is None:
        return None
    return open_task_checkpoint_store(db_path)


def open_default_task_checkpoint_persistence(
    *,
    db_path: Path | None = None,
    implementation: TaskCheckpointPersistence | None = None,
) -> TaskCheckpointPersistence:
    if implementation is not None:
        return implementation
    path = db_path or resolve_task_checkpoints_db_path(None)
    return open_task_checkpoint_store(path)


def optional_task_checkpoint_reader(
    store: TaskCheckpointPersistence | TaskCheckpointReader | None,
) -> TaskCheckpointReader | None:
    if store is None:
        return None
    return store

