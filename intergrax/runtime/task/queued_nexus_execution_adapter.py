# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RunService adapter that dispatches Nexus tasks through a TaskQueue (§41, J.3)."""

from __future__ import annotations

import time
from typing import Optional

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.service import RunService
from intergrax.queueing.contracts.task_queue import TaskQueue, TaskRequest, TaskStatus
from intergrax.queueing.worker.result_codec import (
    decode_logical_task_result,
    nexus_result_payload_from_envelope,
)
from intergrax.runtime.task.nexus_worker_execution import NexusTaskWorkerOutput
from intergrax.runtime.task.worker_payload import (
    NEXUS_TASK_V2_LOGICAL_NAME,
    encode_execution_request,
)
from intergrax.tools.execution_models import ToolExecutionResult


class QueuedNexusExecutionAdapter(ExecutionAdapter):
    """
    Enqueue Nexus Task v2 execution via Tier-0 TaskQueue (Celery).

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
    ) -> None:
        self._task_queue = task_queue
        self._run_service = run_service
        self._logical_task_name = logical_task_name
        self._wait_for_result = wait_for_result
        self._poll_interval = result_poll_interval_seconds
        self._poll_timeout = result_poll_timeout_seconds

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
            payload = self._decode_worker_output(result.output)
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

    @staticmethod
    def _decode_worker_output(raw: object) -> Optional[dict]:
        if isinstance(raw, bytes):
            try:
                envelope = decode_logical_task_result(raw)
            except (UnicodeDecodeError, ValueError, TypeError):
                return None
            payload = nexus_result_payload_from_envelope(envelope)
            if payload is not None:
                return payload
        if isinstance(raw, ToolExecutionResult):
            if not raw.success or raw.output is None:
                return None
            if isinstance(raw.output, NexusTaskWorkerOutput):
                return dict(raw.output.result_payload)
            if hasattr(raw.output, "result_payload"):
                return dict(raw.output.result_payload)
            return None
        if isinstance(raw, dict):
            if "result_payload" in raw:
                return dict(raw["result_payload"])
            payload = nexus_result_payload_from_envelope(raw)
            if payload is not None:
                return payload
            output = raw.get("output")
            if isinstance(output, dict) and "result_payload" in output:
                return dict(output["result_payload"])
        return None

    def shutdown(self, wait: bool = True) -> None:
        return
