# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest enqueue helper (LKW.4C)."""

from __future__ import annotations

import base64
import json

import pytest

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskRequest, TaskResult, TaskStatus
from intergrax.tools.registry.wiring import ToolWiringContext
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    background_ingest_payload_base64,
    decode_background_ingest_job,
    encode_background_ingest_job,
)
from local_workspace_application.background_ingest.enqueue import (
    build_background_ingest_enqueue_input,
    enqueue_background_ingest_job,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_PAYLOAD_KEYS = frozenset({"content", "chunks", "prompt", "secret"})


def _sample_job(**overrides: object) -> LkwBackgroundIngestJob:
    payload = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "source_paths": ("/data/user_docs/proof.txt",),
    }
    payload.update(overrides)
    return LkwBackgroundIngestJob.model_validate(payload)


class _FakeMessageBus(TaskQueue):
    def __init__(self) -> None:
        self.requests: list[TaskRequest] = []

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        self.requests.append(request)
        return TaskHandle(task_id="task-1", provider="fake", tenant_id=request.tenant_id)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return TaskStatus.PENDING

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        return None


def test_builder_creates_valid_message_bus_enqueue_input() -> None:
    job = _sample_job(run_id="run-1")
    params = build_background_ingest_enqueue_input(job)

    assert params.tenant_id == job.tenant_id
    assert params.run_id == "run-1"
    assert params.task_name == LKW_BACKGROUND_INGEST_TASK_NAME
    assert params.payload_base64 == background_ingest_payload_base64(job)
    assert params.idempotency_key == background_ingest_idempotency_key(job)


def test_explicit_run_id_override_wins() -> None:
    job = _sample_job(run_id="job-run")
    params = build_background_ingest_enqueue_input(job, run_id="override-run")

    assert params.run_id == "override-run"


def test_missing_job_run_id_uses_deterministic_fallback() -> None:
    job = _sample_job(run_id=None)
    first = build_background_ingest_enqueue_input(job)
    second = build_background_ingest_enqueue_input(job)

    assert first.run_id
    assert first.run_id == background_ingest_idempotency_key(job)
    assert second.run_id == first.run_id


def test_builder_payload_is_accepted_and_decodable() -> None:
    job = _sample_job(run_id="run-1")
    params = build_background_ingest_enqueue_input(job)

    decoded = decode_background_ingest_job(base64.b64decode(params.payload_base64))

    assert decoded == job


def test_service_calls_platform_message_bus_enqueue_path() -> None:
    job = _sample_job(run_id="run-1")
    bus = _FakeMessageBus()
    ctx = ToolWiringContext(message_bus=bus)

    output = enqueue_background_ingest_job(ctx, job, run_id="run-1")

    assert output.task_id == "task-1"
    assert output.provider == "fake"
    assert output.tenant_id == job.tenant_id
    assert len(bus.requests) == 1
    request = bus.requests[0]
    assert request.tenant_id == job.tenant_id
    assert request.run_id == "run-1"
    assert request.task_name == LKW_BACKGROUND_INGEST_TASK_NAME
    assert request.payload == encode_background_ingest_job(job)
    assert request.idempotency_key == background_ingest_idempotency_key(job)


def test_service_fails_when_message_bus_not_configured() -> None:
    job = _sample_job(run_id="run-1")
    ctx = ToolWiringContext(message_bus=None)

    with pytest.raises(RuntimeError, match="message_bus_not_configured"):
        enqueue_background_ingest_job(ctx, job)


def test_enqueue_input_payload_excludes_raw_content_like_fields() -> None:
    job = _sample_job(run_id="run-1", reason="incremental", priority="high")
    params = build_background_ingest_enqueue_input(job)
    payload = json.loads(base64.b64decode(params.payload_base64).decode("utf-8"))

    assert _FORBIDDEN_PAYLOAD_KEYS.isdisjoint(payload.keys())
