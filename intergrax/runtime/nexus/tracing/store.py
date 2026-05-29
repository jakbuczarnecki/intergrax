# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Trace store path resolution and SQLite opener (§33, §42.24)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore

ENV_TRACE_DB = "INTERGRAX_TRACE_DB"
DEFAULT_TRACE_DB = Path("build") / "intergrax_trace.db"


def resolve_trace_db_path(explicit: Path | str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_TRACE_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_TRACE_DB


def open_run_trace_store(db_path: Path | None = None) -> SQLiteRunTraceStore:
    """Open (and create) the default SQLite-backed run trace store."""
    path = db_path or resolve_trace_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteRunTraceStore(db_path=path)
