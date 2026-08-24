# © Artur Czarnecki. All rights reserved.

"""Execution-identity evidence tests for LKW file watcher (P0-1)."""

from __future__ import annotations

import json

from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from local_workspace_application.background_ingest.contracts import (
    background_ingest_idempotency_key,
)
from local_workspace_application.file_watcher.batching import (
    build_file_watcher_ingest_job,
    build_incremental_file_change_batch,
    file_change_token,
)
from local_workspace_application.file_watcher.contracts import FileChange, FileSnapshot
from local_workspace_application.file_watcher.execution_evidence import (
    BackgroundIngestExecutionIdentityEvidence,
    FileWatcherIngestEnqueuedRecord,
    extract_file_watcher_ingest_enqueued_records,
    validate_background_ingest_execution_identity_evidence,
)
from local_workspace_application.file_watcher.runtime import FileWatcherCycleResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SAMPLE_IDEMPOTENCY = "lkw.background_ingest.v1:0123456789abcdef0123456789abcdef"
_SAMPLE_CHANGE_TOKEN = "sha256:" + ("a" * 64)


def _sample_enqueued_record() -> FileWatcherIngestEnqueuedRecord:
    return FileWatcherIngestEnqueuedRecord(
        change_token=_SAMPLE_CHANGE_TOKEN,
        task_id=_SAMPLE_IDEMPOTENCY,
        provider="kafka",
        tenant_id="tenant-a",
        broker_run_id=_SAMPLE_IDEMPOTENCY,
        idempotency_key=_SAMPLE_IDEMPOTENCY,
    )


def test_runtime_enqueued_cycle_exposes_broker_run_id_and_idempotency_key(
    tmp_path,
) -> None:
    path = str(tmp_path / "doc.txt")
    snap = FileSnapshot(path=path, size_bytes=1, modified_time_ns=1)
    batch = build_incremental_file_change_batch(
        (FileChange(path=path, kind="created", current=snap),)
    )
    job = build_file_watcher_ingest_job(
        batch,
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        collection_id="collection-a",
        allowed_roots=frozenset({str(tmp_path.resolve())}),
    )
    expected_key = background_ingest_idempotency_key(job)
    result = FileWatcherCycleResult(
        status="enqueued",
        actionable_path_count=1,
        change_token=batch.change_token,
        task_id=expected_key,
        provider="kafka",
        tenant_id="tenant-a",
        broker_run_id=expected_key,
        idempotency_key=expected_key,
    )
    assert result.broker_run_id == expected_key
    assert result.idempotency_key == expected_key


def test_extract_watcher_ingest_enqueued_records_from_logs() -> None:
    record = _sample_enqueued_record()
    logs = "\n".join(
        [
            "noise",
            record.model_dump_json(),
            '{"schema_version":"other"}',
        ]
    )
    extracted = extract_file_watcher_ingest_enqueued_records(logs)
    assert len(extracted) == 1
    assert extracted[0] == record


def test_validate_execution_identity_rejects_mismatched_task_id() -> None:
    watcher = _sample_enqueued_record()
    other_key = "lkw.background_ingest.v1:ffffffffffffffffffffffffffffffff"
    worker = BackgroundIngestExecutionIdentityEvidence(
        message_bus_task_id=other_key,
        broker_run_id=other_key,
        idempotency_key=other_key,
        change_token=watcher.change_token,
        runtime_task_id=mint_task_id(),
        runtime_run_id=mint_run_id(),
    )
    with pytest.raises(ValueError, match="message_bus_task_id_mismatch"):
        validate_background_ingest_execution_identity_evidence(
            watcher_record=watcher,
            worker_evidence=worker,
        )


def test_validate_execution_identity_rejects_unrelated_change_token(tmp_path: Path) -> None:
    watcher = _sample_enqueued_record()
    other_path = str((tmp_path / "other.txt").resolve())
    worker = BackgroundIngestExecutionIdentityEvidence(
        message_bus_task_id=watcher.task_id,
        broker_run_id=watcher.broker_run_id,
        idempotency_key=watcher.idempotency_key,
        change_token=file_change_token(
            (FileSnapshot(path=other_path, size_bytes=2, modified_time_ns=2),)
        )
        or "sha256:" + ("b" * 64),
        runtime_task_id=mint_task_id(),
        runtime_run_id=mint_run_id(),
    )
    with pytest.raises(ValueError, match="change_token_mismatch"):
        validate_background_ingest_execution_identity_evidence(
            watcher_record=watcher,
            worker_evidence=worker,
        )


def test_validate_execution_identity_accepts_linked_evidence() -> None:
    watcher = _sample_enqueued_record()
    worker = BackgroundIngestExecutionIdentityEvidence(
        message_bus_task_id=watcher.task_id,
        broker_run_id=watcher.broker_run_id,
        idempotency_key=watcher.idempotency_key,
        change_token=watcher.change_token,
        runtime_task_id=mint_task_id(),
        runtime_run_id=mint_run_id(),
    )
    validate_background_ingest_execution_identity_evidence(
        watcher_record=watcher,
        worker_evidence=worker,
    )


def test_missing_runtime_run_id_rejected() -> None:
    with pytest.raises(ValidationError):
        BackgroundIngestExecutionIdentityEvidence(
            message_bus_task_id=_SAMPLE_IDEMPOTENCY,
            broker_run_id=_SAMPLE_IDEMPOTENCY,
            idempotency_key=_SAMPLE_IDEMPOTENCY,
            change_token=_SAMPLE_CHANGE_TOKEN,
            runtime_task_id=mint_task_id(),
            runtime_run_id="not-canonical",
        )


def test_enqueued_cycle_requires_matching_broker_and_idempotency_keys() -> None:
    with pytest.raises(ValidationError):
        FileWatcherCycleResult(
            status="enqueued",
            actionable_path_count=1,
            change_token=_SAMPLE_CHANGE_TOKEN,
            task_id=_SAMPLE_IDEMPOTENCY,
            provider="kafka",
            tenant_id="tenant-a",
            broker_run_id=_SAMPLE_IDEMPOTENCY,
            idempotency_key="lkw.background_ingest.v1:ffffffffffffffffffffffffffffffff",
        )


def test_ingest_enqueued_record_roundtrip_json() -> None:
    record = _sample_enqueued_record()
    parsed = json.loads(record.model_dump_json())
    restored = FileWatcherIngestEnqueuedRecord.model_validate(parsed)
    assert restored == record
