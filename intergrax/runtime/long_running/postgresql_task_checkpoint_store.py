# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PostgreSQL checkpoint store for resumable tasks (DIST-SCALE-X P4-A / SX-F01)."""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta
from typing import Any, List, Mapping, Optional

from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLSession,
    is_postgresql_unique_violation,
)
from intergrax.runtime.long_running.checkpoint_revision import (
    CheckpointIdConflictError,
    CheckpointRevisionRequiredError,
    StaleCheckpointWriteError,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeCancellationError,
    ScheduledResumeClaim,
    ScheduledResumeScheduleConflictError,
    SchedulerActionClaim,
)
from intergrax.runtime.long_running.scheduled_resume_metadata import (
    ScheduledResumeMetadataValidationError,
)
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumeStatus,
    validate_scheduled_resume_for_persistence,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome, ExecutionTerminalRecord
from intergrax.runtime.task.task_state import TaskState

__all__ = ["PostgreSQLTaskCheckpointStore"]

_PAUSED_TASK_STATES = (
    TaskState.WAITING_FOR_HUMAN.value,
    TaskState.WAITING_FOR_RESOURCES.value,
    TaskState.NEEDS_MORE_INFORMATION.value,
)

_SCHEDULER_LEDGER_STARTED = "started"
_SCHEDULER_LEDGER_COMPLETED = "completed"
_SCHEDULER_LEDGER_UNCERTAIN = "uncertain"


class PostgreSQLTaskCheckpointStore(TaskCheckpointPersistence):
    def __init__(
        self,
        *,
        config: PostgreSQLIntegrationConfig | None = None,
        provider: PostgreSQLConnectionProvider | None = None,
    ) -> None:
        if provider is None:
            resolved = config or PostgreSQLIntegrationConfig.from_env()
            provider = PostgreSQLConnectionProvider(resolved)
        self._provider = provider
        self._ensure_schema()

    @staticmethod
    def _stream_advisory_key(tenant_id: str, task_id: str) -> int:
        digest = hashlib.sha256(f"{tenant_id}\0{task_id}".encode()).digest()
        return struct.unpack("!q", digest[:8])[0]

    def _lock_stream(self, session: PostgreSQLSession, tenant_id: str, task_id: str) -> None:
        session.execute(
            "SELECT pg_advisory_xact_lock(%s)",
            (self._stream_advisory_key(tenant_id, task_id),),
        )

    def _ensure_schema(self) -> None:
        with self._provider.transaction() as session:
            session.execute(
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
                    created_at_utc TEXT NOT NULL,
                    runtime_checkpoint_json TEXT,
                    checkpoint_revision INTEGER,
                    store_sequence BIGSERIAL
                );
                """,
            )
            session.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_task
                ON task_checkpoints (task_id, tenant_id, created_at_utc DESC);
                """,
            )
            session.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_token
                ON task_checkpoints (task_id, tenant_id, resume_token);
                """,
            )
            session.execute(
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
                """,
            )
            session.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scheduled_resumes_due
                ON scheduled_resumes (status, run_at_utc);
                """,
            )
            session.execute(
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
                """,
            )
            session.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_task_checkpoints_stream_revision
                ON task_checkpoints (tenant_id, task_id, checkpoint_revision);
                """,
            )
            session.execute(
                """
                CREATE TABLE IF NOT EXISTS task_execution_terminal (
                    tenant_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    run_id TEXT,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    recorded_at_utc TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, task_id)
                );
                """,
            )

    def get_terminal_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
        with self._provider.connection() as session:
            row = session.execute(
                """
                SELECT tenant_id, task_id, run_id, outcome, reason, recorded_at_utc
                FROM task_execution_terminal
                WHERE tenant_id = %s AND task_id = %s
                """,
                (tenant_id, task_id),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_terminal_record(row)

    def put_terminal_record_if_absent(self, record: ExecutionTerminalRecord) -> bool:
        with self._provider.transaction() as session:
            inserted = session.execute(
                """
                INSERT INTO task_execution_terminal (
                    tenant_id, task_id, run_id, outcome, reason, recorded_at_utc
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, task_id) DO NOTHING
                """,
                (
                    record.tenant_id,
                    record.task_id,
                    str(record.run_id) if record.run_id is not None else None,
                    record.outcome.value,
                    record.reason,
                    record.recorded_at_utc,
                ),
            )
            return inserted.rowcount == 1

    @staticmethod
    def _checkpoint_write_payload(checkpoint: TaskCheckpoint) -> tuple[object, ...]:
        runtime_payload = (
            json.dumps(checkpoint.runtime.model_dump(mode="json"), sort_keys=True)
            if checkpoint.runtime is not None
            else None
        )
        return (
            checkpoint.task_id,
            checkpoint.tenant_id,
            checkpoint.resume_token,
            checkpoint.task_state.value,
            json.dumps(checkpoint.task_snapshot, sort_keys=True),
            checkpoint.progress_message,
            checkpoint.notify_channel,
            checkpoint.created_at_utc,
            checkpoint.schema_version,
            runtime_payload,
        )

    def save(
        self,
        checkpoint: TaskCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> TaskCheckpoint:
        with self._provider.transaction() as session:
            self._lock_stream(session, checkpoint.tenant_id, checkpoint.task_id)
            existing = session.execute(
                """
                SELECT store_sequence, checkpoint_id, task_id, tenant_id, resume_token,
                       task_state, task_snapshot_json, progress_message, notify_channel,
                       created_at_utc, runtime_checkpoint_json, checkpoint_revision
                FROM task_checkpoints
                WHERE checkpoint_id = %s
                """,
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if existing is not None:
                existing_checkpoint = self._row_to_checkpoint(existing)
                if self._checkpoint_write_payload(existing_checkpoint) != self._checkpoint_write_payload(
                    checkpoint,
                ):
                    raise CheckpointIdConflictError(
                        checkpoint_id=checkpoint.checkpoint_id,
                        task_id=checkpoint.task_id,
                        tenant_id=checkpoint.tenant_id,
                    )
                return existing_checkpoint

            current_row = session.execute(
                """
                SELECT checkpoint_revision FROM task_checkpoints
                WHERE tenant_id = %s AND task_id = %s
                ORDER BY checkpoint_revision DESC NULLS LAST
                LIMIT 1
                """,
                (checkpoint.tenant_id, checkpoint.task_id),
            ).fetchone()
            current_revision = (
                int(current_row["checkpoint_revision"])
                if current_row is not None and current_row["checkpoint_revision"] is not None
                else None
            )
            if current_revision is None:
                if expected_revision is not None:
                    raise StaleCheckpointWriteError(
                        task_id=checkpoint.task_id,
                        tenant_id=checkpoint.tenant_id,
                        expected_revision=expected_revision,
                        actual_revision=None,
                    )
                next_revision = 1
            else:
                if expected_revision is None:
                    raise CheckpointRevisionRequiredError(
                        task_id=checkpoint.task_id,
                        tenant_id=checkpoint.tenant_id,
                        actual_revision=current_revision,
                    )
                if expected_revision != current_revision:
                    raise StaleCheckpointWriteError(
                        task_id=checkpoint.task_id,
                        tenant_id=checkpoint.tenant_id,
                        expected_revision=expected_revision,
                        actual_revision=current_revision,
                    )
                next_revision = current_revision + 1

            inserted = session.execute(
                """
                INSERT INTO task_checkpoints (
                    checkpoint_id, task_id, tenant_id, resume_token, task_state,
                    task_snapshot_json, progress_message, notify_channel, created_at_utc,
                    runtime_checkpoint_json, checkpoint_revision
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING store_sequence
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
                    next_revision,
                ),
            ).fetchone()
            if inserted is None:
                raise RuntimeError("checkpoint save did not yield durable store_sequence")
            store_sequence = int(inserted["store_sequence"])
        return checkpoint.model_copy(
            update={"revision": next_revision, "store_sequence": store_sequence},
        )

    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        with self._provider.connection() as session:
            row = session.execute(
                """
                SELECT store_sequence, * FROM task_checkpoints
                WHERE task_id = %s AND tenant_id = %s
                ORDER BY checkpoint_revision DESC
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
        with self._provider.connection() as session:
            row = session.execute(
                """
                SELECT store_sequence, * FROM task_checkpoints
                WHERE task_id = %s AND tenant_id = %s AND resume_token = %s
                ORDER BY checkpoint_revision DESC
                LIMIT 1
                """,
                (task_id, tenant_id, resume_token),
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        with self._provider.connection() as session:
            rows = session.execute(
                """
                SELECT store_sequence, * FROM task_checkpoints
                WHERE task_id = %s AND tenant_id = %s
                ORDER BY checkpoint_revision ASC, store_sequence ASC
                """,
                (task_id, tenant_id),
            ).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def list_paused(self) -> List[TaskCheckpoint]:
        paused_states = _PAUSED_TASK_STATES
        placeholders = ",".join("%s" for _ in paused_states)
        query = f"""
            SELECT c.store_sequence, c.checkpoint_id, c.task_id, c.tenant_id, c.resume_token,
                   c.task_state, c.task_snapshot_json, c.progress_message, c.notify_channel,
                   c.created_at_utc, c.runtime_checkpoint_json, c.checkpoint_revision
            FROM task_checkpoints c
            INNER JOIN (
                SELECT task_id, tenant_id, MAX(checkpoint_revision) AS max_revision
                FROM task_checkpoints
                WHERE task_state IN ({placeholders})
                GROUP BY task_id, tenant_id
            ) latest
            ON c.task_id = latest.task_id
            AND c.tenant_id = latest.tenant_id
            AND c.checkpoint_revision = latest.max_revision
            WHERE c.task_state IN ({placeholders})
        """
        params = paused_states + paused_states
        with self._provider.connection() as session:
            rows = session.execute(query, params).fetchall()
        return [self._row_to_checkpoint(row) for row in rows]

    def schedule(self, entry: ScheduledResume) -> ScheduledResume:
        validated = validate_scheduled_resume_for_persistence(entry)
        with self._provider.transaction() as session:
            try:
                session.execute(
                    """
                    INSERT INTO scheduled_resumes (
                        schedule_id, task_id, tenant_id, resume_token, run_at_utc,
                        status, resume_metadata_json, created_at_utc
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        validated.schedule_id,
                        validated.task_id,
                        validated.tenant_id,
                        validated.resume_token,
                        validated.run_at_utc,
                        validated.status.value,
                        json.dumps(validated.resume_metadata),
                        validated.created_at_utc,
                    ),
                )
            except Exception as exc:
                if is_postgresql_unique_violation(exc):
                    raise ScheduledResumeScheduleConflictError(
                        f"Duplicate schedule_id rejected: {validated.schedule_id!r}",
                    ) from exc
                raise
        return validated

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
        with self._provider.transaction() as session:
            session.execute(
                """
                UPDATE scheduled_resumes
                SET status = %s
                WHERE status = %s
                  AND lease_expires_at_utc IS NOT NULL
                  AND lease_expires_at_utc < %s
                """,
                (
                    ScheduledResumeStatus.UNCERTAIN.value,
                    ScheduledResumeStatus.RUNNING.value,
                    now.isoformat(),
                ),
            )
            rows = session.execute(
                """
                SELECT schedule_id FROM scheduled_resumes
                WHERE status = %s AND run_at_utc <= %s
                ORDER BY run_at_utc ASC, schedule_id ASC
                LIMIT %s
                """,
                (ScheduledResumeStatus.PENDING.value, before_utc_iso, limit),
            ).fetchall()
            for row in rows:
                schedule_id = row["schedule_id"]
                fence_row = session.execute(
                    "SELECT fence FROM scheduled_resumes WHERE schedule_id = %s",
                    (schedule_id,),
                ).fetchone()
                current_fence = int(fence_row["fence"]) if fence_row else 0
                new_fence = current_fence + 1
                updated = session.execute(
                    """
                    UPDATE scheduled_resumes
                    SET status = %s, owner_id = %s, lease_expires_at_utc = %s, fence = %s
                    WHERE schedule_id = %s AND status = %s
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
                full_row = session.execute(
                    "SELECT * FROM scheduled_resumes WHERE schedule_id = %s",
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
        return claims

    def complete_claim(self, claim: ScheduledResumeClaim) -> None:
        with self._provider.transaction() as session:
            updated = session.execute(
                """
                UPDATE scheduled_resumes
                SET status = %s, owner_id = NULL, lease_expires_at_utc = NULL
                WHERE schedule_id = %s
                  AND status = %s
                  AND owner_id = %s
                  AND fence = %s
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
                raise StaleClaimError(
                    f"Stale scheduled resume completion rejected for "
                    f"schedule_id={claim.schedule_id} fence={claim.fence}.",
                )
    def list_due(
        self,
        *,
        before_utc_iso: str,
        limit: int = 100,
    ) -> List[ScheduledResume]:
        with self._provider.connection() as session:
            rows = session.execute(
                """
                SELECT * FROM scheduled_resumes
                WHERE status = %s AND run_at_utc <= %s
                ORDER BY run_at_utc ASC, schedule_id ASC
                LIMIT %s
                """,
                (ScheduledResumeStatus.PENDING.value, before_utc_iso, limit),
            ).fetchall()
        return [self._row_to_scheduled_resume(row) for row in rows]

    def cancel(self, schedule_id: str) -> None:
        with self._provider.transaction() as session:
            row = session.execute(
                "SELECT status FROM scheduled_resumes WHERE schedule_id = %s",
                (schedule_id,),
            ).fetchone()
            if row is None:
                raise ScheduledResumeCancellationError(
                    f"Scheduled resume not found: schedule_id={schedule_id}",
                )
            status = ScheduledResumeStatus(row["status"])
            if status == ScheduledResumeStatus.PENDING:
                session.execute(
                    """
                    UPDATE scheduled_resumes
                    SET status = %s
                    WHERE schedule_id = %s AND status = %s
                    """,
                    (ScheduledResumeStatus.CANCELLED.value, schedule_id, status.value),
                )
                return
            raise ScheduledResumeCancellationError(
                f"Cannot cancel schedule_id={schedule_id} with status={status.value}",
            )

    def has_action(self, ledger_key: str) -> bool:
        with self._provider.connection() as session:
            row = session.execute(
                "SELECT status FROM scheduler_ledger WHERE ledger_key = %s LIMIT 1",
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
        with self._provider.transaction() as session:
            session.execute(
                """
                UPDATE scheduler_ledger
                SET status = %s
                WHERE status = %s
                  AND lease_expires_at_utc IS NOT NULL
                  AND lease_expires_at_utc < %s
                """,
                (
                    _SCHEDULER_LEDGER_UNCERTAIN,
                    _SCHEDULER_LEDGER_STARTED,
                    now.isoformat(),
                ),
            )
            row = session.execute(
                "SELECT status, fence FROM scheduler_ledger WHERE ledger_key = %s",
                (ledger_key,),
            ).fetchone()
            if row is None:
                fence = 1
                session.execute(
                    """
                    INSERT INTO scheduler_ledger (
                        ledger_key, action, executed_at_utc, status,
                        owner_id, lease_expires_at_utc, fence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
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
                return SchedulerActionClaim(
                    ledger_key=ledger_key,
                    action=action,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=fence,
                )

            status = row["status"]
            if status == _SCHEDULER_LEDGER_COMPLETED:
                return None
            if status == _SCHEDULER_LEDGER_UNCERTAIN:
                return None
            if status == _SCHEDULER_LEDGER_STARTED:
                lease_row = session.execute(
                    "SELECT lease_expires_at_utc FROM scheduler_ledger WHERE ledger_key = %s",
                    (ledger_key,),
                ).fetchone()
                lease_raw = lease_row["lease_expires_at_utc"]
                if lease_raw is not None:
                    lease_dt = datetime.fromisoformat(lease_raw)
                    if lease_dt.tzinfo is None:
                        lease_dt = lease_dt.replace(tzinfo=UTC)
                    if lease_dt > now:
                        return None
                session.execute(
                    """
                    UPDATE scheduler_ledger
                    SET status = %s
                    WHERE ledger_key = %s AND status = %s
                    """,
                    (_SCHEDULER_LEDGER_UNCERTAIN, ledger_key, _SCHEDULER_LEDGER_STARTED),
                )
                return None

            return None

    def complete_action(self, claim: SchedulerActionClaim) -> None:
        now = datetime.now(UTC)
        with self._provider.transaction() as session:
            updated = session.execute(
                """
                UPDATE scheduler_ledger
                SET status = %s, executed_at_utc = %s,
                    owner_id = NULL, lease_expires_at_utc = NULL
                WHERE ledger_key = %s
                  AND status = %s
                  AND owner_id = %s
                  AND fence = %s
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
                raise StaleClaimError(
                    f"Stale scheduler action completion rejected for "
                    f"ledger_key={claim.ledger_key} fence={claim.fence}.",
                )
    @staticmethod
    def _row_to_scheduled_resume(row: Mapping[str, Any]) -> ScheduledResume:
        keys = row.keys()
        owner_id = row["owner_id"] if "owner_id" in keys else None
        lease_expires_at_utc = (
            row["lease_expires_at_utc"] if "lease_expires_at_utc" in keys else None
        )
        fence = int(row["fence"]) if "fence" in keys else 0
        try:
            metadata = json.loads(row["resume_metadata_json"])
        except json.JSONDecodeError as exc:
            raise ScheduledResumeMetadataValidationError(
                f"Invalid resume_metadata_json for schedule_id={row['schedule_id']!r}",
            ) from exc
        try:
            return ScheduledResume(
                schedule_id=row["schedule_id"],
                task_id=row["task_id"],
                tenant_id=row["tenant_id"],
                resume_token=row["resume_token"],
                run_at_utc=row["run_at_utc"],
                status=ScheduledResumeStatus(row["status"]),
                resume_metadata=metadata,
                created_at_utc=row["created_at_utc"],
                owner_id=owner_id,
                lease_expires_at_utc=lease_expires_at_utc,
                fence=fence,
            )
        except (ValueError, ScheduledResumeMetadataValidationError) as exc:
            raise ScheduledResumeMetadataValidationError(
                f"Durable scheduled resume row rejected for schedule_id="
                f"{row['schedule_id']!r}: {exc}",
            ) from exc

    @staticmethod
    def _row_to_checkpoint(row: Mapping[str, Any]) -> TaskCheckpoint:
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
            revision=(
                int(row["checkpoint_revision"])
                if "checkpoint_revision" in row.keys() and row["checkpoint_revision"] is not None
                else None
            ),
            store_sequence=int(row["store_sequence"]),
            runtime=runtime,
        )

    @staticmethod
    def _row_to_terminal_record(row: Mapping[str, Any]) -> ExecutionTerminalRecord:
        run_id = row["run_id"]
        return ExecutionTerminalRecord(
            tenant_id=row["tenant_id"],
            task_id=row["task_id"],
            run_id=run_id,
            outcome=ExecutionTerminalOutcome(row["outcome"]),
            reason=row["reason"],
            recorded_at_utc=row["recorded_at_utc"],
        )
