# © Artur Czarnecki. All rights reserved.

"""PCM-SCHEMA-EVOLUTION-INTEGRITY schema migration tests (PCM-07)."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path

import pytest

from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_LEGACY_TASK_CHECKPOINTS_DDL = """
CREATE TABLE task_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    resume_token TEXT NOT NULL,
    task_state TEXT NOT NULL,
    task_snapshot_json TEXT NOT NULL,
    progress_message TEXT NOT NULL DEFAULT '',
    notify_channel TEXT,
    created_at_utc TEXT NOT NULL
);
"""

_LEGACY_CHECKPOINT_ROW = (
    "ckpt_legacy_01",
    "task-legacy",
    "tenant-legacy",
    "resume-token-legacy",
    TaskState.WAITING_FOR_HUMAN.value,
    '{"message": "legacy checkpoint"}',
    "legacy progress",
    "log",
    "2026-08-22T08:00:00+00:00",
)


def _table_columns(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _create_legacy_task_checkpoints_schema(
    db_path: Path,
    *,
    with_row: bool = False,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(_LEGACY_TASK_CHECKPOINTS_DDL)
        if with_row:
            conn.execute(
                """
                INSERT INTO task_checkpoints (
                    checkpoint_id, task_id, tenant_id, resume_token, task_state,
                    task_snapshot_json, progress_message, notify_channel, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _LEGACY_CHECKPOINT_ROW,
            )


def test_new_database_adds_runtime_checkpoint_json_column(tmp_path: Path) -> None:
    db_path = tmp_path / "ckpt.db"
    SQLiteTaskCheckpointStore(db_path=db_path)
    assert "runtime_checkpoint_json" in _table_columns(db_path, "task_checkpoints")


def test_old_schema_migrates_runtime_checkpoint_json_column(tmp_path: Path) -> None:
    db_path = tmp_path / "ckpt.db"
    _create_legacy_task_checkpoints_schema(db_path)
    assert "runtime_checkpoint_json" not in _table_columns(db_path, "task_checkpoints")

    SQLiteTaskCheckpointStore(db_path=db_path)
    assert "runtime_checkpoint_json" in _table_columns(db_path, "task_checkpoints")


def test_second_initialization_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "ckpt.db"
    SQLiteTaskCheckpointStore(db_path=db_path)
    SQLiteTaskCheckpointStore(db_path=db_path)
    assert "runtime_checkpoint_json" in _table_columns(db_path, "task_checkpoints")


def test_unexpected_operational_error_propagates(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ckpt.db"
    _create_legacy_task_checkpoints_schema(db_path)

    class _InstrumentedConnection:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql, parameters=(), /):
            if (
                isinstance(sql, str)
                and "runtime_checkpoint_json" in sql
                and sql.strip().upper().startswith("ALTER")
            ):
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, parameters)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self._conn.close()
            return False

    def _patched_connection(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return _InstrumentedConnection(conn)

    monkeypatch.setattr(SQLiteTaskCheckpointStore, "_connection", _patched_connection)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        SQLiteTaskCheckpointStore(db_path=db_path)


def test_existing_checkpoint_row_preserved_after_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "ckpt.db"
    _create_legacy_task_checkpoints_schema(db_path, with_row=True)

    store = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store.get_by_token("task-legacy", "tenant-legacy", "resume-token-legacy")

    assert loaded is not None
    assert loaded.checkpoint_id == "ckpt_legacy_01"
    assert loaded.progress_message == "legacy progress"
    assert loaded.task_snapshot == {"message": "legacy checkpoint"}
    assert loaded.runtime is None


def test_no_catch_all_operational_error_for_runtime_checkpoint_json_migration() -> None:
    source = inspect.getsource(SQLiteTaskCheckpointStore._ensure_schema)
    assert "runtime_checkpoint_json" in source
    assert "except sqlite3.OperationalError" not in source
