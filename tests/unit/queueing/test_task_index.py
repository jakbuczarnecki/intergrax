# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import time

import pytest

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus, TaskSummary
from intergrax.queueing.providers.rabbitmq.rabbitmq_task_queue import RabbitMQTaskQueue
from intergrax.queueing.task_index import (
    list_tasks_from_index,
    purge_completed_tasks_from_index,
    record_task_index,
)

pytestmark = pytest.mark.gate


class InMemoryKVStore(DistributedKVStore):
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
        del ttl_seconds
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
        del ttl_seconds
        current = self._data.get((tenant_id, key))
        if current != expected:
            return False
        self._data[(tenant_id, key)] = new_value
        return True


class _StubProducer:
    def publish(self, *, topic: str, payload: bytes) -> None:
        del topic, payload


def test_purge_completed_removes_terminal_tasks_and_kv_rows() -> None:
    kv = InMemoryKVStore()
    tenant_id = "tenant-a"
    old_epoch = time.time() - 3600
    record_task_index(
        kv,
        tenant_id=tenant_id,
        task_id="done-1",
        task_name="index.rebuild",
        provider="rabbitmq",
        status=TaskStatus.SUCCEEDED,
        updated_at_epoch=old_epoch,
    )
    kv.set(tenant_id, "task:done-1:status", TaskStatus.SUCCEEDED.value.encode("utf-8"))
    kv.set(tenant_id, "task:done-1:result", b"SUCCEEDED|1||")
    record_task_index(
        kv,
        tenant_id=tenant_id,
        task_id="active-1",
        task_name="index.ingest",
        provider="rabbitmq",
        status=TaskStatus.RUNNING,
    )
    kv.set(tenant_id, "task:active-1:status", TaskStatus.RUNNING.value.encode("utf-8"))

    purged = purge_completed_tasks_from_index(
        kv,
        tenant_id=tenant_id,
        provider="rabbitmq",
        older_than_seconds=0,
    )
    assert purged == 1
    assert kv.get(tenant_id, "task:done-1:status") is None
    assert kv.get(tenant_id, "task:done-1:result") is None
    assert kv.get(tenant_id, "task:active-1:status") is not None

    rows = list_tasks_from_index(kv, tenant_id, provider="rabbitmq", limit=10)
    assert len(rows) == 1
    assert rows[0].task_id == "active-1"


def test_rabbitmq_queue_registers_and_purges_completed() -> None:
    kv = InMemoryKVStore()
    queue = RabbitMQTaskQueue(producer=_StubProducer(), queue="tasks", kv_store=kv)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="run-1",
            task_name="index.rebuild",
            payload=b"{}",
        )
    )
    queue.register_task_index(
        tenant_id="tenant-a",
        task_id=handle.task_id,
        task_name="index.rebuild",
        status=TaskStatus.SUCCEEDED,
    )
    kv.set("tenant-a", queue._status_key(handle.task_id), TaskStatus.SUCCEEDED.value.encode("utf-8"))
    kv.set("tenant-a", queue._result_key(handle.task_id), b"SUCCEEDED|1||")

    rows = queue.list_tasks("tenant-a", limit=10)
    assert len(rows) == 1
    assert isinstance(rows[0], TaskSummary)
    assert rows[0].status == TaskStatus.SUCCEEDED

    purged = queue.purge_completed("tenant-a")
    assert purged == 1
    assert queue.list_tasks("tenant-a", limit=10) == []
