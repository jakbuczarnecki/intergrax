# © Artur Czarnecki. All rights reserved.

"""Unit tests for platform background task registry/runtime."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.events import TaskEvent, TaskEventName
from intergrax.background_tasks.registry import TaskRegistry, UnknownTaskError
from intergrax.background_tasks.worker_runtime import WorkerRuntime
from intergrax.runtime.background_execution.identity_persistence import (
    wire_background_execution_identity_persistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.tools.execution_models import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Output(BaseModel):
    answer: str = "ok"


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
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


class _Collector:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def emit(self, event: TaskEvent) -> None:
        self.events.append(event)


def _handler(
    *,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: str | None = None,
    execution_identity: BackgroundExecutionIdentity,
):
    _ = tenant_id, run_id, payload, idempotency_key, execution_identity
    return ToolExecutionResult.ok(_Output(answer="indexed"))


def test_task_registry_resolves_registered_task() -> None:
    registry = TaskRegistry()
    registry.register(
        TaskDefinition(
            task_name="lkw.background_ingest.v1",
            payload_schema=dict,
            handler=_handler,
        )
    )
    resolved = registry.resolve("lkw.background_ingest.v1")
    assert resolved.task_name == "lkw.background_ingest.v1"


def test_task_registry_rejects_unknown_task_name() -> None:
    registry = TaskRegistry()
    with pytest.raises(UnknownTaskError):
        registry.resolve("missing.task.v1")


def test_worker_runtime_emits_lifecycle_events() -> None:
    kv = _KV()
    task_registry = TaskRegistry()
    task_registry.register(
        TaskDefinition(
            task_name="demo.task.v1",
            payload_schema=dict,
            handler=_handler,
        )
    )
    execution_registry = TaskExecutionRegistry()
    task_registry.bind_execution_registry(execution_registry)
    collector = _Collector()
    from intergrax.background_tasks.state_store import TaskResultStore, TaskStateStore

    runtime = WorkerRuntime(
        registry=task_registry,
        state_store=TaskStateStore(kv_store=kv, provider="kafka"),
        result_store=TaskResultStore(kv_store=kv),
        execution_registry=execution_registry,
        provider="kafka",
        event_emitter=collector,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    request = TaskRequest(
        tenant_id="tenant-a",
        run_id="run-1",
        task_name="demo.task.v1",
        payload=b"{}",
    )
    result = runtime.process_request(request, task_id="run-1")
    assert result.status == TaskStatus.SUCCEEDED
    names = {event.name for event in collector.events}
    assert TaskEventName.DISPATCHED in names
    assert TaskEventName.STARTED in names
    assert TaskEventName.SUCCEEDED in names
    assert TaskEventName.RESULT_STORED in names
