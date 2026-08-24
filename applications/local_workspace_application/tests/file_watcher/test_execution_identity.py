# © Artur Czarnecki. All rights reserved.

"""Execution evidence tests for LKW file watcher (P0-1 / P0-1R1)."""

from __future__ import annotations

import json
import inspect

from pathlib import Path

import pytest
from pydantic import ValidationError

from local_workspace_application.background_ingest.handler import (
    _runtime_result_output,
)
from local_workspace_application.background_ingest.worker_handler import (
    BackgroundIngestWorkerOutput,
)
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
    PLATFORM_EXECUTION_LINKAGE_GAP,
    FileWatcherIngestEnqueuedRecord,
    extract_file_watcher_ingest_enqueued_records,
    validate_file_watcher_enqueue_evidence,
)
from local_workspace_application.file_watcher.runtime import FileWatcherCycleResult
from local_workspace_application.tests.background_ingest.test_background_ingest_handler import (
    _sample_job,
    _sample_request,
)

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


def test_watcher_enqueue_record_requires_message_bus_task_id() -> None:
    with pytest.raises(ValidationError):
        FileWatcherIngestEnqueuedRecord(
            change_token=_SAMPLE_CHANGE_TOKEN,
            task_id="",
            provider="kafka",
            tenant_id="tenant-a",
            broker_run_id=_SAMPLE_IDEMPOTENCY,
            idempotency_key=_SAMPLE_IDEMPOTENCY,
        )


def test_validate_enqueue_evidence_rejects_mismatched_change_token(
    tmp_path: Path,
) -> None:
    watcher = _sample_enqueued_record()
    other_path = str((tmp_path / "other.txt").resolve())
    other_token = file_change_token(
        (FileSnapshot(path=other_path, size_bytes=2, modified_time_ns=2),)
    ) or "sha256:" + ("b" * 64)
    with pytest.raises(ValueError, match="change_token_mismatch"):
        validate_file_watcher_enqueue_evidence(
            watcher,
            expected_change_token=other_token,
        )


def test_validate_enqueue_evidence_accepts_matching_record() -> None:
    watcher = _sample_enqueued_record()
    validate_file_watcher_enqueue_evidence(
        watcher,
        expected_change_token=watcher.change_token,
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


def test_platform_execution_linkage_gap_documented() -> None:
    assert "AttemptId" in PLATFORM_EXECUTION_LINKAGE_GAP
    assert "TaskId" in PLATFORM_EXECUTION_LINKAGE_GAP
    assert "RunId" in PLATFORM_EXECUTION_LINKAGE_GAP


def test_background_ingest_handler_output_has_no_execution_identity_block() -> None:
    source = inspect.getsource(_runtime_result_output)
    assert "execution_identity" not in source


def test_worker_output_has_no_runtime_identity_transport_fields() -> None:
    fields = BackgroundIngestWorkerOutput.model_fields
    assert "runtime_task_id" not in fields
    assert "runtime_run_id" not in fields
    assert "broker_run_id" not in fields
    assert "change_token" not in fields
    assert "idempotency_key" not in fields


def test_handler_queue_payload_excludes_execution_identity_block() -> None:
    from intergrax.runtime.task.task import TaskResult as RuntimeTaskResult, TaskState

    job = _sample_job(change_token=_SAMPLE_CHANGE_TOKEN)
    request = _sample_request(job)
    runtime_result = RuntimeTaskResult(
        task_id="task_runtime_1",
        run_id="run_runtime_1",
        state=TaskState.COMPLETED,
        answer="indexed",
        agent_id="local_indexer",
        metadata={},
    )
    payload = json.loads(
        _runtime_result_output(runtime_result, request=request, job=job).decode("utf-8")
    )
    assert "execution_identity" not in payload
