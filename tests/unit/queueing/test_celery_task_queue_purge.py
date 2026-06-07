# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
from intergrax.queueing.task_index import list_tasks_from_index

pytestmark = pytest.mark.unit


class MemoryKV:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}

    def get(self, *, tenant_id: str, key: str) -> bytes | None:
        return self._data.get((tenant_id, key))

    def set(self, *, tenant_id: str, key: str, value: bytes) -> None:
        self._data[(tenant_id, key)] = value

    def delete(self, *, tenant_id: str, key: str) -> None:
        self._data.pop((tenant_id, key), None)


def test_celery_purge_completed_uses_task_index() -> None:
    app = MagicMock()
    app.tasks = {"intergrax.execute": MagicMock()}
    result = MagicMock()
    result.id = "celery-task-1"
    app.tasks["intergrax.execute"].apply_async.return_value = result

    kv = MemoryKV()
    queue = CeleryTaskQueue(app, kv_store=kv)
    handle = queue.enqueue(
        TaskRequest(
            tenant_id="tenant-1",
            run_id="run-1",
            task_name="demo",
            payload=b"{}",
        )
    )
    assert handle.task_id == "celery-task-1"

    async_result = MagicMock()
    async_result.state = "SUCCESS"
    async_result.result = b"ok"
    async_result.retries = 0
    app.AsyncResult.return_value = async_result

    queue.get_result(handle)
    rows = list_tasks_from_index(kv, "tenant-1", provider="celery")
    assert rows[0].status == TaskStatus.SUCCEEDED

    purged = queue.purge_completed("tenant-1")
    assert purged == 1
    assert list_tasks_from_index(kv, "tenant-1", provider="celery") == []
