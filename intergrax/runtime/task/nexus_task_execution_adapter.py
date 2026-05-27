# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.service import RunService
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task_run_bridge import (
    task_from_execution_request,
    task_result_to_payload,
)


class NexusTaskExecutionAdapter(ExecutionAdapter):
    """
    Bridges FastAPI Core RunService to NexusLoop (Phase A.1).

    ExecutionRequest.run_id becomes Task.task_id for unified trace correlation.
    """

    def __init__(self, nexus_loop: NexusLoop) -> None:
        self._nexus_loop = nexus_loop
        self._run_service: Optional[RunService] = None

    def bind_run_service(self, run_service: RunService) -> None:
        self._run_service = run_service

    async def start_execution(self, request: ExecutionRequest) -> None:
        if self._run_service is None:
            raise RuntimeError(
                "NexusTaskExecutionAdapter.run_service not bound. "
                "Call bind_run_service() after DefaultRunService construction."
            )

        run_id = request.run_id
        self._run_service.mark_running(run_id)

        try:
            task = task_from_execution_request(request)
            result = await self._nexus_loop.handle_task(task)
            self._run_service.mark_completed(
                run_id,
                result_payload=task_result_to_payload(result),
            )
        except Exception as exc:
            self._run_service.mark_failed(
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def shutdown(self, wait: bool = True) -> None:
        return
