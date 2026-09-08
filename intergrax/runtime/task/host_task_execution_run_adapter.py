# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""FastAPI Core RunService adapter routed through canonical host task execution (NPSC-3G)."""

from __future__ import annotations

from typing import Optional

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.service import RunService
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.interactions.task_executor import HostTaskExecutionExecutor, TaskEnricher
from intergrax.runtime.task.task_run_bridge import (
    task_from_execution_request,
    task_result_to_payload,
)


class HostTaskExecutionRunAdapter(ExecutionAdapter):
    """
    Thin boundary adapter: ExecutionRequest → HostTaskExecutionPort.

    No lifecycle ownership, identity mint, or strategy selection.
    """

    def __init__(
        self,
        host_execution: HostTaskExecutionPort,
        *,
        task_enricher: TaskEnricher | None = None,
    ) -> None:
        self._task_executor = HostTaskExecutionExecutor(
            host_execution,
            task_enricher=task_enricher,
        )
        self._run_service: Optional[RunService] = None

    @property
    def host_execution(self) -> HostTaskExecutionPort:
        return self._task_executor.host_execution

    def bind_run_service(self, run_service: RunService) -> None:
        self._run_service = run_service

    async def start_execution(self, request: ExecutionRequest) -> None:
        if self._run_service is None:
            raise RuntimeError(
                "HostTaskExecutionRunAdapter.run_service not bound. "
                "Call bind_run_service() after DefaultRunService construction."
            )

        run_id = request.run_id
        self._run_service.mark_running(run_id)

        try:
            task = task_from_execution_request(request)
            result = await self._task_executor.execute(task)
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


__all__ = ["HostTaskExecutionRunAdapter"]
