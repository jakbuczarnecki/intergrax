# © Artur Czarnecki. All rights reserved.

"""Durable task row first, optional Celery wake second (ARCH-FREEZE package B)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskSummary,
)
from intergrax.queueing.providers.document_store.document_store_task_queue import (
    DocumentStoreTaskQueue,
)

logger = logging.getLogger(__name__)

WakePublishError = RuntimeError


class DurableWakeTaskQueue(TaskQueue):
    """Authoritative ``DocumentStoreTaskQueue`` with a best-effort wake publisher."""

    def __init__(
        self,
        *,
        durable_queue: DocumentStoreTaskQueue,
        publish_wake: Callable[[TaskHandle, TaskRequest], None] | None = None,
    ) -> None:
        self._durable = durable_queue
        self._publish_wake = publish_wake

    @property
    def durable_queue(self) -> DocumentStoreTaskQueue:
        return self._durable

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        handle = self._durable.enqueue(request)
        if self._publish_wake is None:
            return handle
        try:
            self._publish_wake(handle, request)
        except Exception:
            logger.exception(
                "durable_wake_publish_failed tenant=%s task_id=%s",
                handle.tenant_id,
                handle.task_id,
            )
        return handle

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        return self._durable.get_status(handle)

    def get_status_if_present(self, handle: TaskHandle) -> TaskStatus | None:
        return self._durable.get_status_if_present(handle)

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        return self._durable.get_result(handle)

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        status_filter: TaskStatus | None = None,
    ) -> list[TaskSummary]:
        return self._durable.list_tasks(
            tenant_id,
            limit=limit,
            status_filter=status_filter,
        )

    def cancel(self, handle: TaskHandle) -> bool:
        return self._durable.cancel(handle)

    def enqueue_durable_only(self, request: TaskRequest) -> TaskHandle:
        """Persist the row without publishing a wake message (fail-closed enqueue tests)."""
        return self._durable.enqueue(request)

    def try_publish_wake(self, handle: TaskHandle, request: TaskRequest) -> None:
        if self._publish_wake is None:
            raise WakePublishError("wake publisher is not configured")
        self._publish_wake(handle, request)


__all__ = ["DurableWakeTaskQueue", "WakePublishError"]
