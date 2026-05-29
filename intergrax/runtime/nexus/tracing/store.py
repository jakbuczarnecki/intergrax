# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trace store path resolution and SQLite opener (§33, §42.24)."""

from __future__ import annotations

from pathlib import Path

from intergrax.integrations.providers.sqlite import create_sqlite_trace_store
from intergrax.integrations.providers.sqlite.paths import (
    DEFAULT_TRACE_DB,
    ENV_TRACE_DB,
    resolve_trace_db_path,
)
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore

__all__ = [
    "DEFAULT_TRACE_DB",
    "ENV_TRACE_DB",
    "resolve_trace_db_path",
    "open_run_trace_store",
]


def open_run_trace_store(db_path: Path | None = None) -> SQLiteRunTraceStore:
    """Open SQLite run trace store via ``integrations.providers.sqlite``."""
    if db_path is not None:
        return create_sqlite_trace_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_trace_store()  # type: ignore[return-value]
