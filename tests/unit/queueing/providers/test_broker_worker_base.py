# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import json
import base64
import pytest
from typing import Optional

from pydantic import BaseModel

from intergrax.background_tasks.events import TaskEvent, TaskEventName
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.queueing.task_index import list_tasks_from_index
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.identity_persistence import (
    wire_background_execution_identity_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult


pytestmark = pytest.mark.unit


class InMemoryKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class DummyOutput(BaseModel):
    value: str


class _TestWorker(BrokerWorkerBase):
    def start(self) -> None:
        raise NotImplementedError


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[TaskEvent] = []

    def emit(self, event: TaskEvent) -> None:
        self.events.append(event)


def _build_message(*, task_id: str = "t1", task_name: str = "dummy") -> bytes:
    encoded_payload = base64.b64encode(b"input").decode("ascii")
    message = {
        "task_id": task_id,
        "tenant_id": "tenant-A",
        "run_id": "run-1",
        "task_name": task_name,
        "payload": encoded_payload,
        "idempotency_key": None,
    }
    return json.dumps(message).encode("utf-8")


def test_broker_worker_base_transitions_to_succeeded() -> None:
    kv = InMemoryKVStore()
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[DummyOutput]:
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult(
            success=True,
            output=DummyOutput(value="OK"),
            error=None,
        )

    registry.register("dummy", handler)

    worker = _TestWorker(
        registry=registry,
        kv_store=kv,
        idempotency_store=None,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
    )

    worker.process_message(raw_payload=_build_message())

    status_bytes = kv.get("tenant-A", "task:t1:status")
    assert status_bytes == TaskStatus.SUCCEEDED.value.encode("utf-8")

    result_bytes = kv.get("tenant-A", "task:t1:result")
    assert result_bytes is not None

    decoded = result_bytes.decode("utf-8")
    parts = decoded.split("|", 3)

    assert parts[0] == TaskStatus.SUCCEEDED.value
    assert parts[1] == "1"
    assert parts[2] == ""

    output_decoded = base64.b64decode(parts[3].encode("ascii"))
    assert output_decoded == b'{"value":"OK"}'


def test_broker_worker_base_transitions_to_failed_on_controlled_result() -> None:
    kv = InMemoryKVStore()
    registry = TaskExecutionRegistry()
    collector = _EventCollector()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[DummyOutput]:
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.fail(
            code="index.embedding_failed",
            message="embedding provider unavailable",
        )

    registry.register("dummy", handler)

    worker = _TestWorker(
        registry=registry,
        kv_store=kv,
        idempotency_store=None,
        event_emitter=collector,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
    )

    worker.process_message(raw_payload=_build_message(task_id="t-fail"))

    status_bytes = kv.get("tenant-A", "task:t-fail:status")
    assert status_bytes == TaskStatus.FAILED.value.encode("utf-8")

    result_bytes = kv.get("tenant-A", "task:t-fail:result")
    assert result_bytes is not None
    decoded = result_bytes.decode("utf-8")
    assert decoded == "FAILED|1|index.embedding_failed: embedding provider unavailable|"

    indexed = list_tasks_from_index(kv, "tenant-A", provider="broker")
    assert len(indexed) == 1
    assert indexed[0].task_id == "t-fail"
    assert indexed[0].status == TaskStatus.FAILED.value

    names = [event.name for event in collector.events]
    assert TaskEventName.STARTED in names
    assert TaskEventName.FAILED in names
    assert TaskEventName.SUCCEEDED not in names
    assert TaskEventName.RESULT_STORED not in names


def test_broker_worker_base_transitions_to_failed_on_exception() -> None:
    kv = InMemoryKVStore()
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[DummyOutput]:
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        raise RuntimeError("unexpected worker crash")

    registry.register("dummy", handler)

    worker = _TestWorker(
        registry=registry,
        kv_store=kv,
        idempotency_store=None,
        identity_persistence=wire_background_execution_identity_persistence(kv_store=kv),
    )

    with pytest.raises(RuntimeError, match="unexpected worker crash"):
        worker.process_message(raw_payload=_build_message(task_id="t-exc"))

    status_bytes = kv.get("tenant-A", "task:t-exc:status")
    assert status_bytes == TaskStatus.FAILED.value.encode("utf-8")

    result_bytes = kv.get("tenant-A", "task:t-exc:result")
    assert result_bytes is not None
    decoded = result_bytes.decode("utf-8")
    assert decoded == "FAILED|1|unexpected worker crash|"