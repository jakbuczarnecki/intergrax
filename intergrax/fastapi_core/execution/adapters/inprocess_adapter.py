# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker
from intergrax.fastapi_core.runs.service import RunService


class InProcessExecutionAdapter(ExecutionAdapter):

    def __init__(
        self,
        worker: ExecutionWorker,
        run_service: RunService,
    ) -> None:
        self._worker = worker
        self._run_service = run_service


    async def start_execution(self, request: ExecutionRequest) -> None:
        run_id = request.run_id
        
        self._run_service.mark_running(run_id)
        
        try:
            result_payload = self._worker.execute(request)
            self._run_service.mark_completed(run_id, result_payload=result_payload)
        except Exception as exc:
            self._run_service.mark_failed(
                run_id=run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


    def shutdown(self, wait: bool = True) -> None:
        return
