# © Artur Czarnecki. All rights reserved.

"""Durable compensation job queue (architecture §40.3.3 · ACP-CLOSE-PROD-5)."""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.lease_claim import LeaseOwnership, StaleClaimError
from intergrax.contracts.side_effect import CompensationRequest


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CompensationJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    RETRYABLE = "retryable"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class CompensationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["compensation_job.v1"] = "compensation_job.v1"
    job_id: str = Field(default_factory=lambda: f"cjob_{uuid4().hex}")
    run_id: str
    tenant_id: str
    agent_id: str
    step_index: int = Field(ge=0)
    request: CompensationRequest
    status: CompensationJobStatus = CompensationJobStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    error: str | None = None
    owner_id: str | None = None
    lease_expires_at: datetime | None = None
    fence: int = 0


class CompensationClaim(LeaseOwnership):
    """Ownership token returned by atomic queue claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    idempotency_key: str
    job: CompensationJob


class CompensationQueueStore(ABC):
    """Persistence port for deferred compensation tool invokes."""

    @abstractmethod
    def enqueue(self, job: CompensationJob) -> None: ...

    @abstractmethod
    def get_by_idempotency_key(
        self,
        tenant_id: str,
        idempotency_key: str,
    ) -> CompensationJob | None: ...

    @abstractmethod
    def list_pending(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]: ...

    @abstractmethod
    def list_uncertain(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]: ...

    @abstractmethod
    def claim_pending(
        self,
        tenant_id: str,
        owner_id: str,
        lease_seconds: int,
        *,
        limit: int = 1,
    ) -> list[CompensationClaim]: ...

    @abstractmethod
    def complete_claim(self, claim: CompensationClaim) -> None: ...

    @abstractmethod
    def fail_claim(
        self,
        claim: CompensationClaim,
        error: str,
        *,
        retryable: bool = False,
    ) -> None: ...


def _claimable_statuses(now: datetime) -> tuple[CompensationJobStatus, ...]:
    del now
    return (CompensationJobStatus.PENDING, CompensationJobStatus.RETRYABLE)


def _is_expired_running(job: CompensationJob, now: datetime) -> bool:
    return (
        job.status == CompensationJobStatus.RUNNING
        and job.lease_expires_at is not None
        and job.lease_expires_at <= now
    )


def _job_to_claim(job: CompensationJob) -> CompensationClaim:
    if job.owner_id is None or job.lease_expires_at is None:
        raise RuntimeError("Claimed job missing ownership fields.")
    return CompensationClaim(
        tenant_id=job.tenant_id,
        idempotency_key=job.request.idempotency_key,
        owner_id=job.owner_id,
        lease_expires_at=job.lease_expires_at,
        fence=job.fence,
        job=job,
    )


class InMemoryCompensationQueueStore(CompensationQueueStore):
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], CompensationJob] = {}
        self._lock = threading.Lock()

    def enqueue(self, job: CompensationJob) -> None:
        with self._lock:
            self._jobs[(job.tenant_id, job.request.idempotency_key)] = job

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        idempotency_key: str,
    ) -> CompensationJob | None:
        with self._lock:
            return self._jobs.get((tenant_id, idempotency_key))

    def list_pending(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]:
        with self._lock:
            jobs = [
                job
                for job in self._jobs.values()
                if job.tenant_id == tenant_id
                and job.status in (CompensationJobStatus.PENDING, CompensationJobStatus.RETRYABLE)
            ]
            jobs.sort(key=lambda item: item.created_at)
            return jobs[:limit]

    def list_uncertain(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]:
        with self._lock:
            jobs = [
                job
                for job in self._jobs.values()
                if job.tenant_id == tenant_id
                and job.status == CompensationJobStatus.UNCERTAIN
            ]
            jobs.sort(key=lambda item: item.created_at)
            return jobs[:limit]

    def _quarantine_expired_running(self, *, tenant_id: str, now: datetime) -> None:
        for job in list(self._jobs.values()):
            if job.tenant_id != tenant_id or not _is_expired_running(job, now):
                continue
            key = (job.tenant_id, job.request.idempotency_key)
            self._jobs[key] = job.model_copy(
                update={"status": CompensationJobStatus.UNCERTAIN},
            )

    def _try_claim_one(
        self,
        *,
        tenant_id: str,
        owner_id: str,
        lease_expires_at: datetime,
        now: datetime,
    ) -> CompensationClaim | None:
        candidates = sorted(
            (
                job
                for job in self._jobs.values()
                if job.tenant_id == tenant_id
            ),
            key=lambda item: item.created_at,
        )
        for job in candidates:
            key = (job.tenant_id, job.request.idempotency_key)
            if job.status in _claimable_statuses(now):
                updated = job.model_copy(
                    update={
                        "status": CompensationJobStatus.RUNNING,
                        "owner_id": owner_id,
                        "lease_expires_at": lease_expires_at,
                        "fence": job.fence + 1,
                    },
                )
                self._jobs[key] = updated
                return _job_to_claim(updated)
        return None

    def claim_pending(
        self,
        tenant_id: str,
        owner_id: str,
        lease_seconds: int,
        *,
        limit: int = 1,
    ) -> list[CompensationClaim]:
        now = _utc_now()
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        claims: list[CompensationClaim] = []
        with self._lock:
            self._quarantine_expired_running(tenant_id=tenant_id, now=now)
            while len(claims) < limit:
                claim = self._try_claim_one(
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    now=now,
                )
                if claim is None:
                    break
                claims.append(claim)
        return claims

    def _assert_claim_current(self, claim: CompensationClaim) -> CompensationJob:
        job = self._jobs.get((claim.tenant_id, claim.idempotency_key))
        if job is None:
            raise StaleClaimError("Compensation job no longer exists.")
        if (
            job.owner_id != claim.owner_id
            or job.fence != claim.fence
            or job.status != CompensationJobStatus.RUNNING
        ):
            raise StaleClaimError(
                f"Stale compensation mutation rejected fence={claim.fence}.",
            )
        return job

    def complete_claim(self, claim: CompensationClaim) -> None:
        with self._lock:
            job = self._assert_claim_current(claim)
            self._jobs[(claim.tenant_id, claim.idempotency_key)] = job.model_copy(
                update={"status": CompensationJobStatus.COMPLETED, "error": None},
            )

    def fail_claim(
        self,
        claim: CompensationClaim,
        error: str,
        *,
        retryable: bool = False,
    ) -> None:
        terminal = (
            CompensationJobStatus.RETRYABLE if retryable else CompensationJobStatus.FAILED
        )
        with self._lock:
            job = self._assert_claim_current(claim)
            self._jobs[(claim.tenant_id, claim.idempotency_key)] = job.model_copy(
                update={"status": terminal, "error": error},
            )


class SQLiteCompensationQueueStore(CompensationQueueStore):
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compensation_jobs (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    owner_id TEXT,
                    lease_expires_at TEXT,
                    fence INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant_id, idempotency_key)
                )
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(compensation_jobs)").fetchall()
            }
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE compensation_jobs ADD COLUMN owner_id TEXT")
            if "lease_expires_at" not in columns:
                conn.execute("ALTER TABLE compensation_jobs ADD COLUMN lease_expires_at TEXT")
            if "fence" not in columns:
                conn.execute(
                    "ALTER TABLE compensation_jobs ADD COLUMN fence INTEGER NOT NULL DEFAULT 0",
                )

    def enqueue(self, job: CompensationJob) -> None:
        payload = job.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO compensation_jobs
                    (tenant_id, idempotency_key, payload, status, created_at, fence)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, idempotency_key) DO NOTHING
                """,
                (
                    job.tenant_id,
                    job.request.idempotency_key,
                    json.dumps(payload),
                    job.status.value,
                    job.created_at.isoformat(),
                    job.fence,
                ),
            )

    def _load_row(self, row: sqlite3.Row) -> CompensationJob:
        job = CompensationJob.model_validate(json.loads(row["payload"]))
        return job.model_copy(
            update={
                "status": CompensationJobStatus(row["status"]),
                "owner_id": row["owner_id"],
                "lease_expires_at": (
                    datetime.fromisoformat(row["lease_expires_at"])
                    if row["lease_expires_at"]
                    else None
                ),
                "fence": int(row["fence"]),
            },
        )

    def _load(self, tenant_id: str, idempotency_key: str) -> CompensationJob | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload, status, owner_id, lease_expires_at, fence
                FROM compensation_jobs
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return self._load_row(row)

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        idempotency_key: str,
    ) -> CompensationJob | None:
        return self._load(tenant_id, idempotency_key)

    def list_pending(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload, status, owner_id, lease_expires_at, fence
                FROM compensation_jobs
                WHERE tenant_id = ? AND status IN (?, ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (
                    tenant_id,
                    CompensationJobStatus.PENDING.value,
                    CompensationJobStatus.RETRYABLE.value,
                    limit,
                ),
            ).fetchall()
        return [self._load_row(row) for row in rows]

    def list_uncertain(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload, status, owner_id, lease_expires_at, fence
                FROM compensation_jobs
                WHERE tenant_id = ? AND status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (tenant_id, CompensationJobStatus.UNCERTAIN.value, limit),
            ).fetchall()
        return [self._load_row(row) for row in rows]

    def _quarantine_expired_running(
        self,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        now: datetime,
    ) -> None:
        rows = conn.execute(
            """
            SELECT payload, status, owner_id, lease_expires_at, fence
            FROM compensation_jobs
            WHERE tenant_id = ?
              AND status = ?
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= ?
            """,
            (tenant_id, CompensationJobStatus.RUNNING.value, now.isoformat()),
        ).fetchall()
        for row in rows:
            job = self._load_row(row)
            self._persist_job(
                conn,
                job.model_copy(update={"status": CompensationJobStatus.UNCERTAIN}),
            )

    def _persist_job(self, conn: sqlite3.Connection, job: CompensationJob) -> None:
        payload = job.model_dump(mode="json")
        conn.execute(
            """
            UPDATE compensation_jobs
            SET payload = ?, status = ?, owner_id = ?, lease_expires_at = ?, fence = ?
            WHERE tenant_id = ? AND idempotency_key = ?
            """,
            (
                json.dumps(payload),
                job.status.value,
                job.owner_id,
                job.lease_expires_at.isoformat() if job.lease_expires_at else None,
                job.fence,
                job.tenant_id,
                job.request.idempotency_key,
            ),
        )

    def claim_pending(
        self,
        tenant_id: str,
        owner_id: str,
        lease_seconds: int,
        *,
        limit: int = 1,
    ) -> list[CompensationClaim]:
        now = _utc_now()
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        claims: list[CompensationClaim] = []
        with self._connect() as conn:
            for _ in range(limit):
                conn.execute("BEGIN IMMEDIATE")
                self._quarantine_expired_running(conn, tenant_id=tenant_id, now=now)
                row = conn.execute(
                    """
                    SELECT tenant_id, idempotency_key, payload, status,
                           owner_id, lease_expires_at, fence, created_at
                    FROM compensation_jobs
                    WHERE tenant_id = ?
                      AND status IN (?, ?)
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    (
                        tenant_id,
                        CompensationJobStatus.PENDING.value,
                        CompensationJobStatus.RETRYABLE.value,
                    ),
                ).fetchone()
                if row is None:
                    conn.commit()
                    break

                job = self._load_row(row)
                new_fence = job.fence + 1
                updated = job.model_copy(
                    update={
                        "status": CompensationJobStatus.RUNNING,
                        "owner_id": owner_id,
                        "lease_expires_at": lease_expires_at,
                        "fence": new_fence,
                    },
                )
                changed = conn.execute(
                    """
                    UPDATE compensation_jobs
                    SET payload = ?, status = ?, owner_id = ?, lease_expires_at = ?, fence = ?
                    WHERE tenant_id = ? AND idempotency_key = ? AND fence = ?
                    """,
                    (
                        json.dumps(updated.model_dump(mode="json")),
                        updated.status.value,
                        owner_id,
                        lease_expires_at.isoformat(),
                        new_fence,
                        tenant_id,
                        job.request.idempotency_key,
                        job.fence,
                    ),
                )
                if changed.rowcount != 1:
                    conn.rollback()
                    continue
                conn.commit()
                claims.append(_job_to_claim(updated))
        return claims

    def complete_claim(self, claim: CompensationClaim) -> None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT payload, status, owner_id, lease_expires_at, fence
                FROM compensation_jobs
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (claim.tenant_id, claim.idempotency_key),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise StaleClaimError("Compensation job no longer exists.")
            job = self._load_row(row)
            if (
                job.owner_id != claim.owner_id
                or job.fence != claim.fence
                or job.status != CompensationJobStatus.RUNNING
            ):
                conn.rollback()
                raise StaleClaimError(
                    f"Stale compensation completion rejected fence={claim.fence}.",
                )
            updated = job.model_copy(
                update={"status": CompensationJobStatus.COMPLETED, "error": None},
            )
            self._persist_job(conn, updated)
            conn.commit()

    def fail_claim(
        self,
        claim: CompensationClaim,
        error: str,
        *,
        retryable: bool = False,
    ) -> None:
        terminal = (
            CompensationJobStatus.RETRYABLE if retryable else CompensationJobStatus.FAILED
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT payload, status, owner_id, lease_expires_at, fence
                FROM compensation_jobs
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (claim.tenant_id, claim.idempotency_key),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise StaleClaimError("Compensation job no longer exists.")
            job = self._load_row(row)
            if (
                job.owner_id != claim.owner_id
                or job.fence != claim.fence
                or job.status != CompensationJobStatus.RUNNING
            ):
                conn.rollback()
                raise StaleClaimError(
                    f"Stale compensation failure rejected fence={claim.fence}.",
                )
            updated = job.model_copy(update={"status": terminal, "error": error})
            self._persist_job(conn, updated)
            conn.commit()
