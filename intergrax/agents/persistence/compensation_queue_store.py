# © Artur Czarnecki. All rights reserved.

"""Durable compensation job queue (architecture §40.3.3 · ACP-CLOSE-PROD-5)."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.side_effect import CompensationRequest


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CompensationJobStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


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
    def mark_completed(self, tenant_id: str, idempotency_key: str) -> None: ...

    @abstractmethod
    def mark_failed(self, tenant_id: str, idempotency_key: str, error: str) -> None: ...


class InMemoryCompensationQueueStore(CompensationQueueStore):
    def __init__(self) -> None:
        self._jobs: dict[tuple[str, str], CompensationJob] = {}

    def enqueue(self, job: CompensationJob) -> None:
        self._jobs[(job.tenant_id, job.request.idempotency_key)] = job

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        idempotency_key: str,
    ) -> CompensationJob | None:
        return self._jobs.get((tenant_id, idempotency_key))

    def list_pending(self, tenant_id: str, *, limit: int = 100) -> list[CompensationJob]:
        jobs = [
            job
            for job in self._jobs.values()
            if job.tenant_id == tenant_id and job.status == CompensationJobStatus.PENDING
        ]
        jobs.sort(key=lambda item: item.created_at)
        return jobs[:limit]

    def mark_completed(self, tenant_id: str, idempotency_key: str) -> None:
        job = self.get_by_idempotency_key(tenant_id, idempotency_key)
        if job is None:
            return
        self._jobs[(tenant_id, idempotency_key)] = job.model_copy(
            update={"status": CompensationJobStatus.COMPLETED},
        )

    def mark_failed(self, tenant_id: str, idempotency_key: str, error: str) -> None:
        job = self.get_by_idempotency_key(tenant_id, idempotency_key)
        if job is None:
            return
        self._jobs[(tenant_id, idempotency_key)] = job.model_copy(
            update={"status": CompensationJobStatus.FAILED, "error": error},
        )


class SQLiteCompensationQueueStore(CompensationQueueStore):
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
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
                    PRIMARY KEY (tenant_id, idempotency_key)
                )
                """
            )

    def enqueue(self, job: CompensationJob) -> None:
        payload = job.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO compensation_jobs
                    (tenant_id, idempotency_key, payload, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, idempotency_key) DO NOTHING
                """,
                (
                    job.tenant_id,
                    job.request.idempotency_key,
                    json.dumps(payload),
                    job.status.value,
                    job.created_at.isoformat(),
                ),
            )

    def _load(self, tenant_id: str, idempotency_key: str) -> CompensationJob | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload FROM compensation_jobs
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant_id, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return CompensationJob.model_validate(json.loads(row[0]))

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
                SELECT payload FROM compensation_jobs
                WHERE tenant_id = ? AND status = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (tenant_id, CompensationJobStatus.PENDING.value, limit),
            ).fetchall()
        return [CompensationJob.model_validate(json.loads(row[0])) for row in rows]

    def _update_status(
        self,
        tenant_id: str,
        idempotency_key: str,
        *,
        status: CompensationJobStatus,
        error: str | None = None,
    ) -> None:
        job = self._load(tenant_id, idempotency_key)
        if job is None:
            return
        updated = job.model_copy(update={"status": status, "error": error})
        payload = updated.model_dump(mode="json")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE compensation_jobs
                SET payload = ?, status = ?
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (
                    json.dumps(payload),
                    status.value,
                    tenant_id,
                    idempotency_key,
                ),
            )

    def mark_completed(self, tenant_id: str, idempotency_key: str) -> None:
        self._update_status(
            tenant_id,
            idempotency_key,
            status=CompensationJobStatus.COMPLETED,
        )

    def mark_failed(self, tenant_id: str, idempotency_key: str, error: str) -> None:
        self._update_status(
            tenant_id,
            idempotency_key,
            status=CompensationJobStatus.FAILED,
            error=error,
        )
