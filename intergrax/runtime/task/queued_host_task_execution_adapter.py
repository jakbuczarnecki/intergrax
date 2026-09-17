# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RunService adapter that dispatches host task execution through a TaskQueue (NPSC-3G)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.service import RunService
from intergrax.queueing.contracts.task_queue import TaskQueue, TaskRequest, TaskStatus
from intergrax.queueing.worker.result_codec import decode_host_task_result_payload
from intergrax.runtime.task.worker_payload import (
    NEXUS_TASK_V2_LOGICAL_NAME,
    encode_execution_request,
)

HostTaskResultDecoder = Callable[[object], Optional[dict[str, object]]]


class QueuedHostTaskExecutionAdapter(ExecutionAdapter):
    """
    Enqueue host task execution via Tier-0 TaskQueue (Celery).

    In laboratory/eager mode (``wait_for_result=True``) the adapter blocks until
    the worker returns and then updates RunService — suitable for gate tests and
    single-process deployments.
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        run_service: RunService,
        *,
        logical_task_name: str = NEXUS_TASK_V2_LOGICAL_NAME,
        wait_for_result: bool = False,
        result_poll_interval_seconds: float = 0.05,
        result_poll_timeout_seconds: float = 30.0,
        result_decoder: HostTaskResultDecoder | None = None,
    ) -> None:
        self._task_queue = task_queue
        self._run_service = run_service
        self._logical_task_name = logical_task_name
        self._wait_for_result = wait_for_result
        self._poll_interval = result_poll_interval_seconds
        self._poll_timeout = result_poll_timeout_seconds
        self._decode_worker_result = result_decoder or decode_host_task_result_payload

    async def start_execution(self, request: ExecutionRequest) -> None:
        self._run_service.mark_running(request.run_id)
        try:
            handle = self._task_queue.enqueue(
                TaskRequest(
                    tenant_id=request.tenant_id,
                    run_id=request.run_id,
                    task_name=self._logical_task_name,
                    payload=encode_execution_request(request),
                    idempotency_key=request.run_id,
                )
            )
            if not self._wait_for_result:
                return

            deadline = time.monotonic() + self._poll_timeout
            while time.monotonic() < deadline:
                result = self._task_queue.get_result(handle)
                if result is not None:
                    self._apply_queue_result(request.run_id, result)
                    return
                if self._task_queue.get_status(handle) == TaskStatus.FAILED:
                    result = self._task_queue.get_result(handle)
                    if result is not None:
                        self._apply_queue_result(request.run_id, result)
                    else:
                        self._run_service.mark_failed(
                            request.run_id,
                            error_type="WorkerFailed",
                            error_message="worker task failed without result payload",
                        )
                    return
                time.sleep(self._poll_interval)

            self._run_service.mark_failed(
                request.run_id,
                error_type="WorkerTimeout",
                error_message=f"worker result not available within {self._poll_timeout}s",
            )
        except Exception as exc:
            self._run_service.mark_failed(
                run_id=request.run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _apply_queue_result(self, run_id: str, result) -> None:
        if result.status == TaskStatus.SUCCEEDED and result.output is not None:
            payload = self._decode_worker_result(result.output)
            if payload is None:
                self._run_service.mark_failed(
                    run_id,
                    error_type="WorkerPayloadError",
                    error_message="worker returned empty success payload",
                )
                return
            self._run_service.mark_completed(run_id, result_payload=payload)
            return

        self._run_service.mark_failed(
            run_id,
            error_type="WorkerFailed",
            error_message=result.error_message or "worker task failed",
        )

    def shutdown(self, wait: bool = True) -> None:
        return


__all__ = ["QueuedHostTaskExecutionAdapter", "HostTaskResultDecoder"]
