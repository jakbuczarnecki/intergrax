# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Duck-typed cloud queue wrapper implementing ``TaskQueue`` / ``MessageBus``."""

from __future__ import annotations

from typing import Any, Optional

from intergrax.integrations.contracts.base import HealthStatus
from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskRequest, TaskResult, TaskStatus


class CloudTaskQueue(TaskQueue):
    """Minimal message bus over ``send_message`` / ``receive_message`` / ``get_message`` duck API."""

    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider
        self._results: dict[str, TaskResult] = {}

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        message_id = str(
            self._client.send_message(
                body=request.payload,
                attributes={
                    "tenant_id": request.tenant_id,
                    "run_id": request.run_id,
                    "task_name": request.task_name,
                },
            )
        )
        return TaskHandle(task_id=message_id, provider=self._provider, tenant_id=request.tenant_id)

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        if handle.task_id in self._results:
            return self._results[handle.task_id].status
        status = self._client.get_message_status(handle.task_id)
        mapping = {
            "pending": TaskStatus.PENDING,
            "running": TaskStatus.RUNNING,
            "succeeded": TaskStatus.SUCCEEDED,
            "failed": TaskStatus.FAILED,
        }
        return mapping.get(str(status).lower(), TaskStatus.PENDING)

    def get_result(self, handle: TaskHandle) -> Optional[TaskResult]:
        if handle.task_id in self._results:
            return self._results[handle.task_id]
        raw = self._client.get_message_result(handle.task_id)
        if raw is None:
            return None
        result = TaskResult(status=TaskStatus.SUCCEEDED, output=raw)
        self._results[handle.task_id] = result
        return result

    def health(self) -> HealthStatus:
        from intergrax.integrations._shared.health import probe_client_health

        return probe_client_health(self._client, slug=self._provider)
