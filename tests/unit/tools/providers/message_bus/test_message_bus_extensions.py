# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import Mock

import pytest

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskStatus, TaskSummary
from intergrax.queueing.providers.celery.celery_task_queue import CeleryTaskQueue
from intergrax.tools.providers.message_bus.contracts import (
    MessageBusCancelInput,
    MessageBusListTasksInput,
)
from intergrax.tools.providers.message_bus.service import message_bus_cancel, message_bus_list_tasks
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeMessageBus:
    def __init__(self) -> None:
        self.cancelled: list[TaskHandle] = []
        self._tasks = [
            TaskSummary(
                task_id="task-1",
                tenant_id="t-1",
                task_name="index.rebuild",
                status=TaskStatus.RUNNING,
                provider="celery",
            )
        ]

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        status_filter: TaskStatus | None = None,
    ) -> list[TaskSummary]:
        rows = [item for item in self._tasks if item.tenant_id == tenant_id]
        if status_filter is not None:
            rows = [item for item in rows if item.status == status_filter]
        return rows[:limit]

    def cancel(self, handle: TaskHandle) -> bool:
        self.cancelled.append(handle)
        return True


def test_message_bus_list_tasks() -> None:
    bus = FakeMessageBus()
    ctx = ToolWiringContext(message_bus=bus)
    out = message_bus_list_tasks(ctx, MessageBusListTasksInput(tenant_id="t-1", limit=10))
    assert out.total == 1
    assert out.tasks[0].task_id == "task-1"


def test_message_bus_cancel() -> None:
    bus = FakeMessageBus()
    ctx = ToolWiringContext(message_bus=bus)
    out = message_bus_cancel(
        ctx,
        MessageBusCancelInput(task_id="task-1", provider="celery", tenant_id="t-1"),
    )
    assert out.cancelled is True
    assert bus.cancelled[0].task_id == "task-1"


def test_celery_cancel_revokes_task() -> None:
    mock_app = Mock()
    queue = CeleryTaskQueue(app=mock_app)
    handle = TaskHandle(task_id="task-99", provider="celery")
    assert queue.cancel(handle) is True
    mock_app.control.revoke.assert_called_once_with("task-99", terminate=True)


def test_celery_list_tasks_filters_tenant() -> None:
    mock_app = Mock()
    inspect = Mock()
    inspect.active.return_value = {
        "worker-1": [
            {
                "id": "task-a",
                "name": "intergrax.execute",
                "kwargs": {"tenant_id": "t-1", "logical_task_name": "rag.ingest"},
            },
            {
                "id": "task-b",
                "name": "intergrax.execute",
                "kwargs": {"tenant_id": "t-2", "logical_task_name": "other"},
            },
        ]
    }
    inspect.reserved.return_value = {}
    inspect.scheduled.return_value = {}
    mock_app.control.inspect.return_value = inspect

    queue = CeleryTaskQueue(app=mock_app)
    rows = queue.list_tasks("t-1", limit=10)
    assert len(rows) == 1
    assert rows[0].task_id == "task-a"
    assert rows[0].status == TaskStatus.RUNNING
