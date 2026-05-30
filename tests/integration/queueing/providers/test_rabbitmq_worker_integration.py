# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
import base64
import pytest
from typing import Optional

from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.integrations.providers.rabbitmq.bundle import (
    create_rabbitmq_integration,
    create_rabbitmq_worker,
)
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from pydantic import BaseModel
from intergrax.tools.execution_models import ToolExecutionResult


pytestmark = pytest.mark.integration


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


def test_rabbitmq_worker_end_to_end() -> None:
    queue_name = f"intergrax-test-{uuid.uuid4()}"

    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
    ) -> ToolExecutionResult[DummyOutput]:
        return ToolExecutionResult(
            success=True,
            output=DummyOutput(value="OK"),
            error=None,
        )

    registry.register("dummy", handler)

    kv = InMemoryKVStore()

    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
    )

    worker = create_rabbitmq_worker(
        kv_store=kv,
        execution_registry=registry,
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
        consumer=bundle.consumer,
        poll_timeout_seconds=0.2,
    )

    queue = bundle.message_bus

    request = TaskRequest(
        tenant_id="tenant-A",
        run_id="run-1",
        task_name="dummy",
        payload=b"input",
        idempotency_key=None,
    )

    handle = queue.enqueue(request=request)
    task_id = handle.task_id

    time.sleep(0.5)

    raw = bundle.consumer.poll(timeout_seconds=1.0)
    assert raw is not None

    worker.process_message(raw_payload=raw)

    status_bytes = kv.get("tenant-A", f"task:{task_id}:status")
    assert status_bytes == TaskStatus.SUCCEEDED.value.encode("utf-8")

    result_bytes = kv.get("tenant-A", f"task:{task_id}:result")
    assert result_bytes is not None

    decoded = result_bytes.decode("utf-8")
    parts = decoded.split("|", 3)

    assert parts[0] == TaskStatus.SUCCEEDED.value
    assert parts[1] == "1"
    assert parts[2] == ""

    output_json = base64.b64decode(parts[3].encode("ascii"))
    output_model = DummyOutput.model_validate_json(output_json.decode("utf-8"))

    assert output_model.value == "OK"
