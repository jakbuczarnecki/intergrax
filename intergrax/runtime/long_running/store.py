# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite checkpoint store for resumable tasks (Phase F.4)."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    DEFAULT_TASK_CHECKPOINTS_DB,
    ENV_TASK_CHECKPOINTS_DB,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeCancellationError,
    ScheduledResumeClaim,
    SchedulerActionClaim,
)
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumeStatus,
)
from intergrax.runtime.task.task import TaskState

__all__ = [
    "DEFAULT_TASK_CHECKPOINTS_DB",
    "ENV_TASK_CHECKPOINTS_DB",
    "SQLiteTaskCheckpointStore",
    "resolve_task_checkpoints_db_path",
]

_PAUSED_TASK_STATES = (
    TaskState.WAITING_FOR_HUMAN.value,
    TaskState.WAITING_FOR_RESOURCES.value,
    TaskState.NEEDS_MORE_INFORMATION.value,
)

_SCHEDULER_LEDGER_STARTED = "started"
_SCHEDULER_LEDGER_COMPLETED = "completed"
_SCHEDULER_LEDGER_UNCERTAIN = "uncertain"


def resolve_task_checkpoints_db_path(explicit: Path | None = None) -> Path:
    from intergrax.integrations.providers.relational_store.sqlite.paths import (
        resolve_task_checkpoints_db_path as _resolve,
    )

    return _resolve(explicit)


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
            checkpoint_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(task_checkpoints)").fetchall()
            }
            if "runtime_checkpoint_json" not in checkpoint_columns:
                conn.execute(
                    "ALTER TABLE task_checkpoints ADD COLUMN runtime_checkpoint_json TEXT",
                )
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
                    created_at_utc TEXT NOT NULL,
                    owner_id TEXT,
                    lease_expires_at_utc TEXT,
                    fence INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            scheduled_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(scheduled_resumes)").fetchall()
            }
            if "owner_id" not in scheduled_columns:
                conn.execute("ALTER TABLE scheduled_resumes ADD COLUMN owner_id TEXT")
            if "lease_expires_at_utc" not in scheduled_columns:
                conn.execute(
                    "ALTER TABLE scheduled_resumes ADD COLUMN lease_expires_at_utc TEXT",
                )
            if "fence" not in scheduled_columns:
                conn.execute(
                    "ALTER TABLE scheduled_resumes ADD COLUMN fence INTEGER NOT NULL DEFAULT 0",
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
                    executed_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    owner_id TEXT,
                    lease_expires_at_utc TEXT,
                    fence INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            ledger_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(scheduler_ledger)").fetchall()
            }
            if "status" not in ledger_columns:
                conn.execute(
                    "ALTER TABLE scheduler_ledger ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'",
                )
            if "owner_id" not in ledger_columns:
                conn.execute("ALTER TABLE scheduler_ledger ADD COLUMN owner_id TEXT")
            if "lease_expires_at_utc" not in ledger_columns:
                conn.execute(
                    "ALTER TABLE scheduler_ledger ADD COLUMN lease_expires_at_utc TEXT",
                )
            if "fence" not in ledger_columns:
                conn.execute(
                    "ALTER TABLE scheduler_ledger ADD COLUMN fence INTEGER NOT NULL DEFAULT 0",
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

    def claim_due(
        self,
        *,
        before_utc_iso: str,
        owner_id: str,
        lease_seconds: int,
        limit: int = 100,
    ) -> List[ScheduledResumeClaim]:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        claims: List[ScheduledResumeClaim] = []
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE scheduled_resumes
                SET status = ?
                WHERE status = ?
                  AND lease_expires_at_utc IS NOT NULL
                  AND lease_expires_at_utc < ?
                """,
                (
                    ScheduledResumeStatus.UNCERTAIN.value,
                    ScheduledResumeStatus.RUNNING.value,
                    now.isoformat(),
                ),
            )
            rows = conn.execute(
                """
                SELECT schedule_id FROM scheduled_resumes
                WHERE status = ? AND run_at_utc <= ?
                ORDER BY run_at_utc ASC
                LIMIT ?
                """,
                (ScheduledResumeStatus.PENDING.value, before_utc_iso, limit),
            ).fetchall()
            for row in rows:
                schedule_id = row["schedule_id"]
                fence_row = conn.execute(
                    "SELECT fence FROM scheduled_resumes WHERE schedule_id = ?",
                    (schedule_id,),
                ).fetchone()
                current_fence = int(fence_row["fence"]) if fence_row else 0
                new_fence = current_fence + 1
                updated = conn.execute(
                    """
                    UPDATE scheduled_resumes
                    SET status = ?, owner_id = ?, lease_expires_at_utc = ?, fence = ?
                    WHERE schedule_id = ? AND status = ?
                    """,
                    (
                        ScheduledResumeStatus.RUNNING.value,
                        owner_id,
                        lease_expires_at.isoformat(),
                        new_fence,
                        schedule_id,
                        ScheduledResumeStatus.PENDING.value,
                    ),
                )
                if updated.rowcount != 1:
                    continue
                full_row = conn.execute(
                    "SELECT * FROM scheduled_resumes WHERE schedule_id = ?",
                    (schedule_id,),
                ).fetchone()
                entry = self._row_to_scheduled_resume(full_row)
                claims.append(
                    ScheduledResumeClaim(
                        schedule_id=schedule_id,
                        owner_id=owner_id,
                        lease_expires_at=lease_expires_at,
                        fence=new_fence,
                        entry=entry,
                    ),
                )
            conn.commit()
        return claims

    def complete_claim(self, claim: ScheduledResumeClaim) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE scheduled_resumes
                SET status = ?, owner_id = NULL, lease_expires_at_utc = NULL
                WHERE schedule_id = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                """,
                (
                    ScheduledResumeStatus.COMPLETED.value,
                    claim.schedule_id,
                    ScheduledResumeStatus.RUNNING.value,
                    claim.owner_id,
                    claim.fence,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise StaleClaimError(
                    f"Stale scheduled resume completion rejected for "
                    f"schedule_id={claim.schedule_id} fence={claim.fence}.",
                )
            conn.commit()

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

    def cancel(self, schedule_id: str) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM scheduled_resumes WHERE schedule_id = ?",
                (schedule_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ScheduledResumeCancellationError(
                    f"Scheduled resume not found: schedule_id={schedule_id}",
                )
            status = ScheduledResumeStatus(row["status"])
            if status == ScheduledResumeStatus.PENDING:
                conn.execute(
                    """
                    UPDATE scheduled_resumes
                    SET status = ?
                    WHERE schedule_id = ? AND status = ?
                    """,
                    (ScheduledResumeStatus.CANCELLED.value, schedule_id, status.value),
                )
                conn.commit()
                return
            conn.rollback()
            raise ScheduledResumeCancellationError(
                f"Cannot cancel schedule_id={schedule_id} with status={status.value}",
            )

    def has_action(self, ledger_key: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status FROM scheduler_ledger WHERE ledger_key = ? LIMIT 1",
                (ledger_key,),
            ).fetchone()
        if row is None:
            return False
        return row["status"] == _SCHEDULER_LEDGER_COMPLETED

    def claim_action(
        self,
        ledger_key: str,
        owner_id: str,
        lease_seconds: int,
        *,
        action: str,
    ) -> Optional[SchedulerActionClaim]:
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE scheduler_ledger
                SET status = ?
                WHERE status = ?
                  AND lease_expires_at_utc IS NOT NULL
                  AND lease_expires_at_utc < ?
                """,
                (
                    _SCHEDULER_LEDGER_UNCERTAIN,
                    _SCHEDULER_LEDGER_STARTED,
                    now.isoformat(),
                ),
            )
            row = conn.execute(
                "SELECT status, fence FROM scheduler_ledger WHERE ledger_key = ?",
                (ledger_key,),
            ).fetchone()
            if row is None:
                fence = 1
                conn.execute(
                    """
                    INSERT INTO scheduler_ledger (
                        ledger_key, action, executed_at_utc, status,
                        owner_id, lease_expires_at_utc, fence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_key,
                        action,
                        now.isoformat(),
                        _SCHEDULER_LEDGER_STARTED,
                        owner_id,
                        lease_expires_at.isoformat(),
                        fence,
                    ),
                )
                conn.commit()
                return SchedulerActionClaim(
                    ledger_key=ledger_key,
                    action=action,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=fence,
                )

            status = row["status"]
            if status == _SCHEDULER_LEDGER_COMPLETED:
                conn.rollback()
                return None
            if status == _SCHEDULER_LEDGER_UNCERTAIN:
                conn.commit()
                return None
            if status == _SCHEDULER_LEDGER_STARTED:
                lease_row = conn.execute(
                    "SELECT lease_expires_at_utc FROM scheduler_ledger WHERE ledger_key = ?",
                    (ledger_key,),
                ).fetchone()
                lease_raw = lease_row["lease_expires_at_utc"]
                if lease_raw is not None:
                    lease_dt = datetime.fromisoformat(lease_raw)
                    if lease_dt.tzinfo is None:
                        lease_dt = lease_dt.replace(tzinfo=UTC)
                    if lease_dt > now:
                        conn.rollback()
                        return None
                conn.execute(
                    """
                    UPDATE scheduler_ledger
                    SET status = ?
                    WHERE ledger_key = ? AND status = ?
                    """,
                    (_SCHEDULER_LEDGER_UNCERTAIN, ledger_key, _SCHEDULER_LEDGER_STARTED),
                )
                conn.commit()
                return None

            conn.rollback()
            return None

    def complete_action(self, claim: SchedulerActionClaim) -> None:
        now = datetime.now(UTC)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE scheduler_ledger
                SET status = ?, executed_at_utc = ?,
                    owner_id = NULL, lease_expires_at_utc = NULL
                WHERE ledger_key = ?
                  AND status = ?
                  AND owner_id = ?
                  AND fence = ?
                """,
                (
                    _SCHEDULER_LEDGER_COMPLETED,
                    now.isoformat(),
                    claim.ledger_key,
                    _SCHEDULER_LEDGER_STARTED,
                    claim.owner_id,
                    claim.fence,
                ),
            )
            if updated.rowcount != 1:
                conn.rollback()
                raise StaleClaimError(
                    f"Stale scheduler action completion rejected for "
                    f"ledger_key={claim.ledger_key} fence={claim.fence}.",
                )
            conn.commit()

    @staticmethod
    def _row_to_scheduled_resume(row: sqlite3.Row) -> ScheduledResume:
        keys = row.keys()
        owner_id = row["owner_id"] if "owner_id" in keys else None
        lease_expires_at_utc = (
            row["lease_expires_at_utc"] if "lease_expires_at_utc" in keys else None
        )
        fence = int(row["fence"]) if "fence" in keys else 0
        return ScheduledResume(
            schedule_id=row["schedule_id"],
            task_id=row["task_id"],
            tenant_id=row["tenant_id"],
            resume_token=row["resume_token"],
            run_at_utc=row["run_at_utc"],
            status=ScheduledResumeStatus(row["status"]),
            resume_metadata=json.loads(row["resume_metadata_json"]),
            created_at_utc=row["created_at_utc"],
            owner_id=owner_id,
            lease_expires_at_utc=lease_expires_at_utc,
            fence=fence,
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
