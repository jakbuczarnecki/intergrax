# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Run trace store factories — Execution Engine owner zone."""

from __future__ import annotations

from pathlib import Path

from intergrax.contracts.run_trace_store import RunTraceStore
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.nexus.tracing.store import open_run_trace_store, resolve_trace_db_path as _resolve_trace_db_path


def resolve_trace_db_path(db_path: Path | None) -> Path:
    return _resolve_trace_db_path(db_path)


def create_in_memory_run_trace_store() -> RunTraceStore:
    return InMemoryRunTraceStore()


def create_sqlite_run_trace_store(db_path: Path | None = None) -> RunTraceStore:
    path = db_path or resolve_trace_db_path(None)
    return SQLiteRunTraceStore(path)


def open_host_run_trace_store(db_path: Path | None = None) -> RunTraceStore:
    return open_run_trace_store(db_path or resolve_trace_db_path(None))


__all__ = [
    "create_in_memory_run_trace_store",
    "create_sqlite_run_trace_store",
    "open_host_run_trace_store",
    "resolve_trace_db_path",
]
