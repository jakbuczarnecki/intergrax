# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve trace store path from env or CLI flag."""

from __future__ import annotations

import os
from pathlib import Path

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
