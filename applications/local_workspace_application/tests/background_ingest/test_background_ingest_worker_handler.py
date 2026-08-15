# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest worker handler registration (LKW.4E)."""

from __future__ import annotations

import json

import pytest

from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.task.task import Task, TaskResult as RuntimeTaskResult, TaskState
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_TASK_NAME,
    encode_background_ingest_job,
)
from local_workspace_application.background_ingest.worker_handler import (
    register_background_ingest_worker_handler,
)
from local_workspace_application.tests.background_ingest.test_background_ingest_handler import (
    _sample_job,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeRunner:
    async def run_task(self, task: Task) -> RuntimeTaskResult:
        return RuntimeTaskResult(
            task_id=task.task_id,
            run_id=task.task_id,
            state=TaskState.COMPLETED,
            answer="indexed",
            agent_id=task.agent_id,
            metadata={"ingest_summary": {"used": True, "reason": "ingest_complete"}},
        )


def test_worker_handler_registers_and_returns_tool_execution_result() -> None:
    registry = TaskExecutionRegistry()
    register_background_ingest_worker_handler(registry, _FakeRunner())
    job = _sample_job()
    handler = registry.get_handler(LKW_BACKGROUND_INGEST_TASK_NAME)

    result = handler(
        tenant_id=job.tenant_id,
        run_id="run-worker-1",
        payload=encode_background_ingest_job(job),
        idempotency_key=None,
    )

    assert result.success is True
    assert result.output is not None
    assert result.output.answer == "indexed"
    assert result.output.schema_version == "lkw.background_ingest_worker.v1"


def test_worker_handler_returns_failure_for_invalid_payload() -> None:
    registry = TaskExecutionRegistry()
    register_background_ingest_worker_handler(registry, _FakeRunner())
    handler = registry.get_handler(LKW_BACKGROUND_INGEST_TASK_NAME)

    result = handler(
        tenant_id="tenant-a",
        run_id="run-worker-2",
        payload=b"not-json",
        idempotency_key=None,
    )

    assert result.success is False
    assert result.error is not None
