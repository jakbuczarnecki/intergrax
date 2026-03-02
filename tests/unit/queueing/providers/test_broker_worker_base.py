# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import json
import base64
import pytest
from typing import Optional

from pydantic import BaseModel

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskStatus
from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
from intergrax.queueing.worker.registry import TaskExecutionRegistry
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


def test_broker_worker_base_transitions_to_succeeded() -> None:
    kv = InMemoryKVStore()
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

    worker = _TestWorker(
        registry=registry,
        kv_store=kv,
        idempotency_store=None,
    )

    encoded_payload = base64.b64encode(b"input").decode("ascii")

    message = {
        "task_id": "t1",
        "tenant_id": "tenant-A",
        "run_id": "run-1",
        "task_name": "dummy",
        "payload": encoded_payload,
        "idempotency_key": None,
    }

    raw = json.dumps(message).encode("utf-8")

    worker.process_message(raw_payload=raw)

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