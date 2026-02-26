# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Optional
from unittest.mock import Mock

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.worker.execution import execute_logical_task
from intergrax.queueing.worker.registry import TaskExecutionRegistry


class DummyKVStore(DistributedKVStore):
    def __init__(self) -> None:
        self.storage = {}

    def get(self, tenant_id: str, key: str) -> Optional[bytes]:
        return self.storage.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.storage[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        self.storage.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: Optional[bytes],
        new_value: bytes,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        current = self.storage.get((tenant_id, key))
        if current == expected:
            self.storage[(tenant_id, key)] = new_value
            return True
        return False


def test_execute_without_idempotency() -> None:
    registry = TaskExecutionRegistry()
    handler = Mock(return_value=b"ok")
    registry.register("task.a", handler)

    result = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key=None,
        kv_store=None,
        lock_ttl_seconds=None,
    )

    assert result == b"ok"
    handler.assert_called_once()


def test_execute_with_idempotency_fresh() -> None:
    registry = TaskExecutionRegistry()
    handler = Mock(return_value=b"fresh")
    registry.register("task.a", handler)

    kv = DummyKVStore()

    result = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="abc",
        kv_store=kv,
        lock_ttl_seconds=60,
    )

    assert result == b"fresh"
    handler.assert_called_once()


def test_execute_with_existing_result() -> None:
    registry = TaskExecutionRegistry()
    handler = Mock(return_value=b"should_not_run")
    registry.register("task.a", handler)

    kv = DummyKVStore()
    kv.storage[("t1", "idempotency:t1:abc")] = b"cached"

    result = execute_logical_task(
        registry=registry,
        logical_task_name="task.a",
        tenant_id="t1",
        run_id="r1",
        payload=b"data",
        idempotency_key="abc",
        kv_store=kv,
        lock_ttl_seconds=60,
    )

    assert result == b"cached"
    handler.assert_not_called()


def test_execute_lock_held() -> None:
    registry = TaskExecutionRegistry()
    handler = Mock(return_value=b"x")
    registry.register("task.a", handler)

    kv = DummyKVStore()
    kv.storage[("t1", "idempotency:t1:abc")] = b"__LOCK__"

    with pytest.raises(RuntimeError):
        execute_logical_task(
            registry=registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="abc",
            kv_store=kv,
            lock_ttl_seconds=60,
        )


def test_missing_ttl_raises() -> None:
    registry = TaskExecutionRegistry()
    handler = Mock(return_value=b"x")
    registry.register("task.a", handler)

    kv = DummyKVStore()

    with pytest.raises(ValueError):
        execute_logical_task(
            registry=registry,
            logical_task_name="task.a",
            tenant_id="t1",
            run_id="r1",
            payload=b"data",
            idempotency_key="abc",
            kv_store=kv,
            lock_ttl_seconds=None,
        )