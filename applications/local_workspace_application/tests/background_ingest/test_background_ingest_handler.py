# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest worker handler contract (LKW.4D)."""

from __future__ import annotations

import json

import pytest

from intergrax.contracts.execution_identity import validate_task_id
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.tools.registry.wiring import ToolWiringContext  # noqa: F401 — primes import graph
from intergrax.runtime.task.task import Task, TaskResult as RuntimeTaskResult, TaskState
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    encode_background_ingest_job,
)
from local_workspace_application.background_ingest.handler import (
    LKW_BACKGROUND_INGEST_AGENT_ID,
    LKW_BACKGROUND_INGEST_CAPABILITY,
    build_background_ingest_runtime_task,
    decode_background_ingest_task_request,
    handle_background_ingest_task_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_METADATA_KEYS = frozenset({"content", "chunks", "prompt", "secret"})


def _sample_job(**overrides: object) -> LkwBackgroundIngestJob:
    payload = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "source_paths": ("/data/user_docs/proof.txt",),
    }
    payload.update(overrides)
    return LkwBackgroundIngestJob.model_validate(payload)


def _sample_request(
    job: LkwBackgroundIngestJob,
    *,
    run_id: str = "lkw.background_ingest.v1:abc123",
    tenant_id: str | None = None,
    task_name: str = LKW_BACKGROUND_INGEST_TASK_NAME,
    payload: bytes | None = None,
) -> TaskRequest:
    return TaskRequest(
        tenant_id=tenant_id if tenant_id is not None else job.tenant_id,
        run_id=run_id,
        task_name=task_name,
        payload=payload if payload is not None else encode_background_ingest_job(job),
        idempotency_key=background_ingest_idempotency_key(job),
    )


class _FakeRunner:
    def __init__(self) -> None:
        self.tasks: list[Task] = []
        self.raise_error: Exception | None = None
        self.ingest_used = True
        self.ingest_reason = "ingest_complete"

    async def run_task(self, task: Task) -> RuntimeTaskResult:
        if self.raise_error is not None:
            raise self.raise_error
        self.tasks.append(task)
        return RuntimeTaskResult(
            task_id=task.task_id,
            run_id=task.task_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id=task.agent_id,
            metadata={
                "ingest_summary": {
                    "used": self.ingest_used,
                    "reason": self.ingest_reason,
                }
            },
        )


def test_decode_helper_accepts_valid_task_request() -> None:
    job = _sample_job(run_id="run-1", correlation_id="corr-1", reason="batch")
    request = _sample_request(job, run_id="run-1")

    decoded = decode_background_ingest_task_request(request)

    assert decoded == job


def test_decode_helper_rejects_wrong_task_name() -> None:
    job = _sample_job()
    request = _sample_request(job, task_name="other.task.v1")

    with pytest.raises(ValueError, match="unexpected_background_ingest_task_name"):
        decode_background_ingest_task_request(request)


def test_decode_helper_rejects_tenant_mismatch() -> None:
    job = _sample_job()
    request = _sample_request(job, tenant_id="tenant-b")

    with pytest.raises(ValueError, match="background_ingest_tenant_mismatch"):
        decode_background_ingest_task_request(request)


def test_runtime_task_builder_maps_job_to_local_workspace_index_task() -> None:
    broker_run_id = "lkw.background_ingest.v1:abc123"
    job = _sample_job(
        requested_by="watcher",
        correlation_id="corr-1",
        reason="batch",
        priority="high",
        change_token="sha256:" + ("a" * 64),
    )
    request = _sample_request(job, run_id=broker_run_id)

    task = build_background_ingest_runtime_task(request, job)

    assert task.task_id.startswith("task_")
    validate_task_id(task.task_id)
    assert task.task_id != request.run_id
    assert task.metadata["background_ingest_broker_run_id"] == broker_run_id
    assert task.metadata["background_ingest_change_token"] == job.change_token
    assert task.tenant_id == job.tenant_id
    assert task.user_id == job.requested_by
    assert task.agent_id == LKW_BACKGROUND_INGEST_AGENT_ID
    assert task.context.capability == LKW_BACKGROUND_INGEST_CAPABILITY
    assert task.metadata["source_paths"] == list(job.source_paths)
    assert task.metadata["collection_id"] == job.collection_id
    assert task.metadata["workspace_id"] == job.workspace_id
    assert task.metadata["tenant_id"] == job.tenant_id
    assert task.metadata["chunking_strategy_id"] == "recursive"
    assert task.metadata["background_ingest_task_name"] == LKW_BACKGROUND_INGEST_TASK_NAME
    assert task.metadata["background_ingest_idempotency_key"] == request.idempotency_key
    assert task.metadata["background_ingest_priority"] == job.priority
    assert task.metadata["background_ingest_correlation_id"] == job.correlation_id
    assert task.metadata["background_ingest_reason"] == job.reason


def test_runtime_task_builder_excludes_raw_content_like_fields() -> None:
    job = _sample_job()
    request = _sample_request(job)

    task = build_background_ingest_runtime_task(request, job)

    assert _FORBIDDEN_METADATA_KEYS.isdisjoint(task.metadata.keys())


@pytest.mark.asyncio
async def test_handler_output_includes_execution_identity_block() -> None:
    job = _sample_job(change_token="sha256:" + ("a" * 64))
    request = _sample_request(job)
    runner = _FakeRunner()

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.SUCCEEDED
    assert queue_result.output is not None
    decoded_output = json.loads(queue_result.output.decode("utf-8"))
    identity = decoded_output.get("execution_identity")
    assert isinstance(identity, dict)
    assert identity.get("runtime_task_id") == runner.tasks[0].task_id
    assert identity.get("broker_run_id") == request.run_id
    assert identity.get("idempotency_key") == request.idempotency_key
    assert identity.get("change_token") == job.change_token


@pytest.mark.asyncio
async def test_handler_delegates_to_runner_and_returns_queue_success_result() -> None:
    job = _sample_job()
    request = _sample_request(job)
    runner = _FakeRunner()

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.SUCCEEDED
    assert queue_result.output is not None
    decoded_output = json.loads(queue_result.output.decode("utf-8"))
    assert decoded_output["answer"] == "indexed"
    assert len(runner.tasks) == 1
    assert runner.tasks[0].context.capability == LKW_BACKGROUND_INGEST_CAPABILITY
    validate_task_id(runner.tasks[0].task_id)
    assert runner.tasks[0].metadata["background_ingest_broker_run_id"] == request.run_id


@pytest.mark.asyncio
async def test_handler_reaches_runner_without_task_id_validation_error() -> None:
    job = _sample_job()
    request = _sample_request(job, run_id="lkw.background_ingest.v1:deadbeef")
    runner = _FakeRunner()

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.SUCCEEDED
    assert len(runner.tasks) == 1
    task = runner.tasks[0]
    validate_task_id(task.task_id)
    assert task.task_id != request.run_id


@pytest.mark.asyncio
async def test_handler_returns_failed_when_request_task_name_is_wrong() -> None:
    job = _sample_job()
    request = _sample_request(job, task_name="other.task.v1")
    runner = _FakeRunner()

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.FAILED
    assert queue_result.error_message is not None
    assert "unexpected_background_ingest_task_name" in queue_result.error_message
    assert runner.tasks == []


@pytest.mark.asyncio
async def test_handler_returns_failed_when_payload_is_invalid() -> None:
    job = _sample_job()
    request = _sample_request(job, payload=b"not-json")
    runner = _FakeRunner()

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.FAILED
    assert queue_result.error_message is not None
    assert runner.tasks == []


@pytest.mark.asyncio
async def test_handler_returns_failed_when_ingest_did_not_index() -> None:
    job = _sample_job()
    request = _sample_request(job)
    runner = _FakeRunner()
    runner.ingest_used = False
    runner.ingest_reason = "source_not_found"

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.FAILED
    assert queue_result.error_message == "source_not_found"


@pytest.mark.asyncio
async def test_handler_returns_failed_when_runner_raises() -> None:
    job = _sample_job()
    request = _sample_request(job)
    runner = _FakeRunner()
    runner.raise_error = RuntimeError("runner_failed")

    queue_result = await handle_background_ingest_task_request(request, runner)

    assert queue_result.status == TaskStatus.FAILED
    assert queue_result.error_message is not None
    assert "runner_failed" in queue_result.error_message
    assert queue_result.attempts == 1
