# © Artur Czarnecki. All rights reserved.

"""Structured execution evidence for LKW file-watcher enqueue (P0-1 / P0-1R1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FILE_WATCHER_INGEST_ENQUEUED_SCHEMA = "lkw.file_watcher_ingest_enqueued.v1"

PLATFORM_EXECUTION_LINKAGE_GAP = (
    "No canonical operator-facing platform contract links message-bus task "
    "to runtime TaskId, runtime RunId, or runtime AttemptId for this workload."
)


class FileWatcherIngestEnqueuedRecord(BaseModel):
    """Safe structured watcher enqueue evidence (no paths or payloads)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["lkw.file_watcher_ingest_enqueued.v1"] = (
        FILE_WATCHER_INGEST_ENQUEUED_SCHEMA
    )
    change_token: str
    task_id: str
    provider: str
    tenant_id: str
    broker_run_id: str
    idempotency_key: str

    @model_validator(mode="after")
    def _validate_identity_fields(self) -> FileWatcherIngestEnqueuedRecord:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.change_token.strip():
            raise ValueError("change_token must be non-empty")
        if not self.broker_run_id.strip():
            raise ValueError("broker_run_id must be non-empty")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        if self.task_id != self.broker_run_id:
            raise ValueError("task_id must equal broker_run_id")
        if self.broker_run_id != self.idempotency_key:
            raise ValueError("broker_run_id must equal idempotency_key")
        return self


def extract_file_watcher_ingest_enqueued_records(
    log_output: str,
) -> tuple[FileWatcherIngestEnqueuedRecord, ...]:
    records: list[FileWatcherIngestEnqueuedRecord] = []
    for line in log_output.splitlines():
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = FileWatcherIngestEnqueuedRecord.model_validate_json(candidate)
        except ValueError:
            continue
        records.append(parsed)
    return tuple(records)


def validate_file_watcher_enqueue_evidence(
    record: FileWatcherIngestEnqueuedRecord,
    *,
    expected_change_token: str | None = None,
) -> None:
    """Fail closed when watcher enqueue evidence is incomplete or contradictory."""
    if not record.task_id.strip():
        raise ValueError("message_bus_task_id_missing")
    if expected_change_token is not None and record.change_token != expected_change_token:
        raise ValueError("change_token_mismatch")
