# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite checkpoint store for resumable tasks (Phase F.4)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.task.task import Task, TaskState
from intergrax.utils.time_provider import SystemTimeProvider

ENV_TASK_CHECKPOINTS_DB = "INTERGRAX_TASK_CHECKPOINTS_DB"
DEFAULT_TASK_CHECKPOINTS_DB = Path("build") / "intergrax_task_checkpoints.db"


def resolve_task_checkpoints_db_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_TASK_CHECKPOINTS_DB, "").strip()
    if env:
        return Path(env)
    return DEFAULT_TASK_CHECKPOINTS_DB


def open_task_checkpoint_store(db_path: Path | None = None) -> SQLiteTaskCheckpointStore:
    path = db_path or resolve_task_checkpoints_db_path(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteTaskCheckpointStore(db_path=path)


class SQLiteTaskCheckpointStore(TaskCheckpointPersistence):
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path
        self._ensure_schema()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_checkpoints (
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
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_task
                ON task_checkpoints (task_id, tenant_id, created_at_utc DESC);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_token
                ON task_checkpoints (task_id, tenant_id, resume_token);
                """
            )
            try:
                conn.execute(
                    "ALTER TABLE task_checkpoints ADD COLUMN runtime_checkpoint_json TEXT"
                )
            except sqlite3.OperationalError:
                pass

    def save(self, checkpoint: TaskCheckpoint) -> TaskCheckpoint:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO task_checkpoints (
                    checkpoint_id, task_id, tenant_id, resume_token, task_state,
                    task_snapshot_json, progress_message, notify_channel, created_at_utc,
                    runtime_checkpoint_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.task_id,
                    checkpoint.tenant_id,
                    checkpoint.resume_token,
                    checkpoint.task_state.value,
                    json.dumps(checkpoint.task_snapshot),
                    checkpoint.progress_message,
                    checkpoint.notify_channel,
                    checkpoint.created_at_utc,
                    json.dumps(checkpoint.runtime.model_dump(mode="json"))
                    if checkpoint.runtime is not None
                    else None,
                ),
            )
        return checkpoint

    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM task_checkpoints
                WHERE task_id = ? AND tenant_id = ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (task_id, tenant_id),
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> Optional[TaskCheckpoint]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM task_checkpoints
                WHERE task_id = ? AND tenant_id = ? AND resume_token = ?
                ORDER BY created_at_utc DESC
                LIMIT 1
                """,
                (task_id, tenant_id, resume_token),
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM task_checkpoints
                WHERE task_id = ? AND tenant_id = ?
                ORDER BY created_at_utc ASC
                """,
                (task_id, tenant_id),
            ).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> TaskCheckpoint:
        runtime_raw = row["runtime_checkpoint_json"] if "runtime_checkpoint_json" in row.keys() else None
        runtime = None
        if runtime_raw:
            runtime = RuntimeCheckpoint.model_validate(json.loads(runtime_raw))
        return TaskCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
            resume_token=row["resume_token"],
            task_state=TaskState(row["task_state"]),
            task_snapshot=json.loads(row["task_snapshot_json"]),
            progress_message=row["progress_message"],
            notify_channel=row["notify_channel"],
            created_at_utc=row["created_at_utc"],
            runtime=runtime,
        )

    @classmethod
    def build_checkpoint(
        cls,
        task: Task,
        *,
        progress_message: str = "",
        resume_token: Optional[str] = None,
        runtime: Optional[RuntimeCheckpoint] = None,
    ) -> TaskCheckpoint:
        token = resume_token or task.runtime.orchestration.resume_token or f"rt_{uuid4().hex[:20]}"
        return TaskCheckpoint(
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            resume_token=token,
            task_state=task.state,
            task_snapshot=task.model_dump(mode="json"),
            progress_message=progress_message,
            notify_channel=task.options.long_running.notify_channel,
            created_at_utc=SystemTimeProvider.utc_now().isoformat(),
            runtime=runtime,
        )
