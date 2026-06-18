# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import List, Optional

from celery import Celery

from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.queueing.contracts.task_queue import (
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskSummary,
)
from intergrax.queueing.task_index import (
    list_tasks_from_index,
    purge_completed_tasks_from_index,
    record_task_index,
    update_task_index_status,
)


class CeleryTaskQueue(TaskQueue):
    """
    Celery-based TaskQueue provider.

    This implementation delegates task submission and result retrieval
    to a provided Celery application instance.

    The Celery application must be configured externally and injected
    during composition phase.
    """

    def __init__(
        self,
        app: Celery,
        *,
        kv_store: DistributedKVStore | None = None,
    ) -> None:
        self._app: Celery = app
        self._kv_store = kv_store


    def enqueue(
        self,
        request: TaskRequest,
    ) -> TaskHandle:
        kwargs = {
            "logical_task_name": request.task_name,
            "tenant_id": request.tenant_id,
            "run_id": request.run_id,
            "payload": request.payload,
            "idempotency_key": request.idempotency_key,
        }
        # Registered task entrypoint — respects task_always_eager (send_task does not).
        result = self._app.tasks["intergrax.execute"].apply_async(kwargs=kwargs)

        handle = TaskHandle(
            task_id=result.id,
            provider="celery",
            tenant_id=request.tenant_id,
        )
        if self._kv_store is not None:
            record_task_index(
                self._kv_store,
                tenant_id=request.tenant_id,
                task_id=handle.task_id,
                task_name=request.task_name,
                provider="celery",
                status=TaskStatus.PENDING,
            )
        return handle


    def get_status(
        self,
        handle: TaskHandle,
    ) -> TaskStatus:
        async_result = self._app.AsyncResult(handle.task_id)

        state = async_result.state

        if state == "PENDING":
            return TaskStatus.PENDING

        if state in ("STARTED", "RETRY"):
            return TaskStatus.RUNNING

        if state == "SUCCESS":
            return TaskStatus.SUCCEEDED

        if state in ("FAILURE", "REVOKED"):
            return TaskStatus.FAILED

        # Fallback: treat unknown states as running
        return TaskStatus.RUNNING


    def get_result(
        self,
        handle: TaskHandle,
    ) -> Optional[TaskResult]:
        async_result = self._app.AsyncResult(handle.task_id)

        state = async_result.state

        if state in ("PENDING", "STARTED", "RETRY"):
            return None

        if state == "SUCCESS":
            from intergrax.queueing.worker.result_codec import worker_result_bytes_from_transport

            if self._kv_store is not None and handle.tenant_id is not None:
                update_task_index_status(
                    self._kv_store,
                    tenant_id=handle.tenant_id,
                    task_id=handle.task_id,
                    provider="celery",
                    status=TaskStatus.SUCCEEDED,
                )
            return TaskResult(
                status=TaskStatus.SUCCEEDED,
                output=worker_result_bytes_from_transport(async_result.result),
                error_message=None,
                attempts=async_result.retries or 0,
            )

        if state in ("FAILURE", "REVOKED"):
            error = async_result.result

            if self._kv_store is not None and handle.tenant_id is not None:
                update_task_index_status(
                    self._kv_store,
                    tenant_id=handle.tenant_id,
                    task_id=handle.task_id,
                    provider="celery",
                    status=TaskStatus.FAILED,
                )
            return TaskResult(
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(error) if error is not None else None,
                attempts=async_result.retries or 0,
            )

        # Fallback — treat unknown states as not finished
        return None

    def cancel(self, handle: TaskHandle) -> bool:
        self._app.control.revoke(handle.task_id, terminate=True)
        return True

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        status_filter: Optional[TaskStatus] = None,
    ) -> List[TaskSummary]:
        inspect = self._app.control.inspect()
        if inspect is None:
            return []

        summaries: list[TaskSummary] = []
        for fetch_name, mapped_status in (
            ("active", TaskStatus.RUNNING),
            ("reserved", TaskStatus.PENDING),
            ("scheduled", TaskStatus.PENDING),
        ):
            fetch = attribute_access.optional(inspect, fetch_name, None)
            if fetch is None:
                continue
            tasks_by_worker = fetch() or {}
            for worker_tasks in tasks_by_worker.values():
                for task in worker_tasks or []:
                    kwargs = task.get("kwargs") or {}
                    if kwargs.get("tenant_id") != tenant_id:
                        continue
                    summary = TaskSummary(
                        task_id=str(task.get("id") or ""),
                        tenant_id=tenant_id,
                        task_name=str(kwargs.get("logical_task_name") or task.get("name") or ""),
                        status=mapped_status,
                        provider="celery",
                    )
                    if status_filter is not None and summary.status != status_filter:
                        continue
                    if summary.task_id:
                        summaries.append(summary)
                    if len(summaries) >= limit:
                        return summaries
        if self._kv_store is not None:
            indexed = list_tasks_from_index(
                self._kv_store,
                tenant_id,
                provider="celery",
                limit=limit,
                status_filter=status_filter,
            )
            if indexed:
                return indexed
        return summaries

    def purge_completed(
        self,
        tenant_id: str,
        *,
        older_than_seconds: int = 0,
    ) -> int:
        if self._kv_store is None:
            return 0
        return purge_completed_tasks_from_index(
            self._kv_store,
            tenant_id=tenant_id,
            provider="celery",
            older_than_seconds=older_than_seconds,
        )