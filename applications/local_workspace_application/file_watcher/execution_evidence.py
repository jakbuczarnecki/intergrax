# © Artur Czarnecki. All rights reserved.

"""Structured execution-identity evidence for LKW file-watcher enqueue (P0-1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.execution_identity import validate_run_id, validate_task_id

FILE_WATCHER_INGEST_ENQUEUED_SCHEMA = "lkw.file_watcher_ingest_enqueued.v1"


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


class BackgroundIngestExecutionIdentityEvidence(BaseModel):
    """Platform execution identity returned through the message-bus result plane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_bus_task_id: str
    broker_run_id: str
    idempotency_key: str
    change_token: str
    runtime_task_id: str
    runtime_run_id: str

    @model_validator(mode="after")
    def _validate_execution_identity(self) -> BackgroundIngestExecutionIdentityEvidence:
        if not self.message_bus_task_id.strip():
            raise ValueError("message_bus_task_id must be non-empty")
        validate_task_id(self.runtime_task_id)
        validate_run_id(self.runtime_run_id)
        if self.message_bus_task_id == self.runtime_task_id:
            raise ValueError("message_bus_task_id must differ from runtime_task_id")
        if self.message_bus_task_id != self.broker_run_id:
            raise ValueError("message_bus_task_id must equal broker_run_id")
        if self.broker_run_id != self.idempotency_key:
            raise ValueError("broker_run_id must equal idempotency_key")
        if not self.change_token.strip():
            raise ValueError("change_token must be non-empty")
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


def validate_background_ingest_execution_identity_evidence(
    *,
    watcher_record: FileWatcherIngestEnqueuedRecord,
    worker_evidence: BackgroundIngestExecutionIdentityEvidence,
) -> None:
    if watcher_record.task_id != worker_evidence.message_bus_task_id:
        raise ValueError("message_bus_task_id_mismatch")
    if watcher_record.change_token != worker_evidence.change_token:
        raise ValueError("change_token_mismatch")
    if watcher_record.broker_run_id != worker_evidence.broker_run_id:
        raise ValueError("broker_run_id_mismatch")
    if watcher_record.idempotency_key != worker_evidence.idempotency_key:
        raise ValueError("idempotency_key_mismatch")
