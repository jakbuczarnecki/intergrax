# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import uuid
import time
import threading
import pytest
from typing import Optional

from pydantic import BaseModel

from intergrax.runtime.transport.config import (
    TransportConfig,
    KafkaTransportConfig,
    RabbitMQTransportConfig,
)
from intergrax.runtime.transport.bootstrap import build_transport

from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.distributed.contracts.kv_store import DistributedKVStore
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


@pytest.mark.parametrize("backend", ["kafka", "rabbitmq"])
def test_transport_concurrency_single_execution(backend: str) -> None:
    queue_name = f"intergrax-concurrency-{uuid.uuid4()}"

    if backend == "kafka":
        config = TransportConfig(
            backend="kafka",
            kafka=KafkaTransportConfig(
                bootstrap_servers="localhost:9092",
            ),
        )
        consumer_group = f"group-{uuid.uuid4()}"
        consumer_group_1 = consumer_group
        consumer_group_2 = consumer_group
    else:
        config = TransportConfig(
            backend="rabbitmq",
            rabbitmq=RabbitMQTransportConfig(
                host="localhost",
                username="intergrax",
                password="intergrax",
            ),
        )
        consumer_group_1 = None
        consumer_group_2 = None

    registry = TaskExecutionRegistry()

    execution_counter = {"count": 0}
    counter_lock = threading.Lock()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
    ) -> ToolExecutionResult[DummyOutput]:
        with counter_lock:
            execution_counter["count"] += 1

        return ToolExecutionResult(
            success=True,
            output=DummyOutput(value="OK"),
            error=None,
        )

    registry.register("dummy", handler)

    kv = InMemoryKVStore()

    # Build two independent transport bundles (two workers)
    bundle1 = build_transport(
        config=config,
        execution_registry=registry,
        kv_store=kv,
        idempotency_store=None,
        queue_name=queue_name,
        consumer_group=consumer_group_1,
    )

    bundle2 = build_transport(
        config=config,
        execution_registry=registry,
        kv_store=kv,
        idempotency_store=None,
        queue_name=queue_name,
        consumer_group=consumer_group_2,
    )

    threads = [
        threading.Thread(target=bundle1.worker.start, daemon=True),
        threading.Thread(target=bundle2.worker.start, daemon=True),
    ]

    for t in threads:
        t.start()

    request = TaskRequest(
        tenant_id="tenant-A",
        run_id="run-1",
        task_name="dummy",
        payload=b"input",
        idempotency_key=None,
    )

    handle = bundle1.task_queue.enqueue(request=request)
    task_id = handle.task_id

    # wait for processing
    deadline = time.time() + 5.0
    while time.time() < deadline:
        status_bytes = kv.get("tenant-A", f"task:{task_id}:status")
        if status_bytes == TaskStatus.SUCCEEDED.value.encode("utf-8"):
            break
        time.sleep(0.1)

    status_bytes = kv.get("tenant-A", f"task:{task_id}:status")
    assert status_bytes == TaskStatus.SUCCEEDED.value.encode("utf-8")

    # Ensure handler executed exactly once
    assert execution_counter["count"] == 1