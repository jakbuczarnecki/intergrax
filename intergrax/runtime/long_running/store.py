# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite checkpoint store for resumable tasks (Phase F.4)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_TASK_CHECKPOINTS_DB,
    ENV_TASK_CHECKPOINTS_DB,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumeStatus,
)
from intergrax.runtime.task.task import Task, TaskState
from intergrax.utils.time_provider import SystemTimeProvider

__all__ = [
    "DEFAULT_TASK_CHECKPOINTS_DB",
    "ENV_TASK_CHECKPOINTS_DB",
    "SQLiteTaskCheckpointStore",
    "resolve_task_checkpoints_db_path",
    "open_task_checkpoint_store",
]

_PAUSED_TASK_STATES = (
    TaskState.WAITING_FOR_HUMAN.value,
    TaskState.WAITING_FOR_RESOURCES.value,
    TaskState.NEEDS_MORE_INFORMATION.value,
)


def resolve_task_checkpoints_db_path(explicit: Path | None = None) -> Path:
    from intergrax.integrations.providers.relational_store.sqlite.paths import (
        resolve_task_checkpoints_db_path as _resolve,
    )

    return _resolve(explicit)


def open_task_checkpoint_store(db_path: Path | None = None) -> SQLiteTaskCheckpointStore:
    from intergrax.integrations.providers.relational_store.sqlite import create_sqlite_task_checkpoint_store

    if db_path is not None:
        return create_sqlite_task_checkpoint_store(db_path=db_path)  # type: ignore[return-value]
    return create_sqlite_task_checkpoint_store()  # type: ignore[return-value]


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_resumes (
                    schedule_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    resume_token TEXT NOT NULL,
                    run_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    resume_metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at_utc TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduled_resumes_due
                ON scheduled_resumes (status, run_at_utc);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduler_ledger (
                    ledger_key TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    executed_at_utc TEXT NOT NULL
                );
                """
            )

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

    def list_paused(self) -> List[TaskCheckpoint]:
        paused_states = _PAUSED_TASK_STATES
        placeholders = ",".join("?" for _ in paused_states)
        query = f"""
            SELECT c.* FROM task_checkpoints c
            INNER JOIN (
                SELECT task_id, tenant_id, MAX(created_at_utc) AS max_created
                FROM task_checkpoints
                WHERE task_state IN ({placeholders})
                GROUP BY task_id, tenant_id
            ) latest
            ON c.task_id = latest.task_id
            AND c.tenant_id = latest.tenant_id
            AND c.created_at_utc = latest.max_created
            WHERE c.task_state IN ({placeholders})
        """
        params = paused_states + paused_states
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_resumes (
                    schedule_id, task_id, tenant_id, resume_token, run_at_utc,
                    status, resume_metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.schedule_id,
                    entry.task_id,
                    entry.tenant_id,
                    entry.resume_token,
                    entry.run_at_utc,
                    entry.status.value,
                    json.dumps(entry.resume_metadata),
                    entry.created_at_utc,
                ),
            )
        return entry

    def list_due(
        self,
        *,
        before_utc_iso: str,
        limit: int = 100,
    ) -> List[ScheduledResume]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scheduled_resumes
                WHERE status = ? AND run_at_utc <= ?
                ORDER BY run_at_utc ASC
                LIMIT ?
                """,
                (ScheduledResumeStatus.PENDING.value, before_utc_iso, limit),
            ).fetchall()
        return [self._row_to_scheduled_resume(row) for row in rows]

    def mark_completed(self, schedule_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scheduled_resumes
                SET status = ?
                WHERE schedule_id = ?
                """,
                (ScheduledResumeStatus.COMPLETED.value, schedule_id),
            )

    def cancel(self, schedule_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scheduled_resumes
                SET status = ?
                WHERE schedule_id = ?
                """,
                (ScheduledResumeStatus.CANCELLED.value, schedule_id),
            )

    def has_action(self, ledger_key: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM scheduler_ledger WHERE ledger_key = ? LIMIT 1",
                (ledger_key,),
            ).fetchone()
        return row is not None

    def record_action(self, ledger_key: str, *, action: str) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduler_ledger (ledger_key, action, executed_at_utc)
                VALUES (?, ?, ?)
                """,
                (ledger_key, action, SystemTimeProvider.utc_now().isoformat()),
            )

    @staticmethod
    def _row_to_scheduled_resume(row: sqlite3.Row) -> ScheduledResume:
        return ScheduledResume(
            schedule_id=row["schedule_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
            resume_token=row["resume_token"],
            run_at_utc=row["run_at_utc"],
            status=ScheduledResumeStatus(row["status"]),
            resume_metadata=json.loads(row["resume_metadata_json"]),
            created_at_utc=row["created_at_utc"],
        )

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
