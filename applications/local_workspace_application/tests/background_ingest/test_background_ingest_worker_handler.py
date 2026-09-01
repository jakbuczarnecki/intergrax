# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest worker handler registration (LKW.4E)."""

from __future__ import annotations

import json

import pytest

from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import bootstrap_background_execution
from intergrax.runtime.background_execution.identity_persistence import (
    KvBackgroundExecutionIdentityPersistence,
)
from intergrax.runtime.background_execution.transport_ref import (
    BackgroundTransportExecutionRef,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
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


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        current = self.get(tenant_id, key)
        if expected is None and current is not None:
            return False
        if expected is not None and current != expected:
            return False
        self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
        return True


def _bootstrap_identity(*, tenant_id: str, transport_task_id: str):
    return bootstrap_background_execution(
        transport_ref=BackgroundTransportExecutionRef(
            tenant_id=tenant_id,
            provider="test",
            transport_task_id=transport_task_id,
        ),
        identity_persistence=KvBackgroundExecutionIdentityPersistence(_KV()),
    )


class _FakeRunner:
    async def run_task(
        self,
        task: Task,
        *,
        run_id: str,
        attempt_id: str,
    ) -> RuntimeTaskResult:
        return RuntimeTaskResult(
            task_id=task.task_id,
            run_id=run_id,
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
        execution_identity=_bootstrap_identity(
            tenant_id=job.tenant_id,
            transport_task_id="ingest-worker-1",
        ),
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
        execution_identity=_bootstrap_identity(
            tenant_id="tenant-a",
            transport_task_id="ingest-worker-2",
        ),
    )

    assert result.success is False
    assert result.error is not None
