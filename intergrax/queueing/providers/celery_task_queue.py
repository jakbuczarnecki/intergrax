# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from celery import Celery

from intergrax.queueing.contracts.task_queue import (
    TaskQueue,
    TaskRequest,
    TaskHandle,
    TaskStatus,
    TaskResult,
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
    ) -> None:
        self._app: Celery = app


    def enqueue(
        self,
        request: TaskRequest,
    ) -> TaskHandle:
        result = self._app.send_task(
            "intergrax.execute",
            kwargs={
                "logical_task_name": request.task_name,
                "tenant_id": request.tenant_id,
                "run_id": request.run_id,
                "payload": request.payload,
                "idempotency_key": request.idempotency_key,
            },
        )

        return TaskHandle(
            task_id=result.id,
            provider="celery",
        )


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
            result = async_result.result

            return TaskResult(
                status=TaskStatus.SUCCEEDED,
                output=result if isinstance(result, bytes) else None,
                error_message=None,
                attempts=async_result.retries or 0,
            )

        if state in ("FAILURE", "REVOKED"):
            error = async_result.result

            return TaskResult(
                status=TaskStatus.FAILED,
                output=None,
                error_message=str(error) if error is not None else None,
                attempts=async_result.retries or 0,
            )

        # Fallback — treat unknown states as not finished
        return None