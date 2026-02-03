# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from intergrax.fastapi_core.execution.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import CancellableExecutionWorker, ExecutionWorker
from intergrax.fastapi_core.runs.service import RunService


class ThreadedExecutionAdapter(ExecutionAdapter):

    def __init__(
        self,
        worker: ExecutionWorker,
        run_service: RunService,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 4,
    ) -> None:
        self._worker = worker
        self._run_service = run_service
        self._futures: dict[str, Future] = {}

        if executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False


    async def start_execution(self, request: ExecutionRequest) -> None:
        run_id = request.run_id

        # boundary: lifecycle start
        self._run_service.mark_running(run_id)

        def _run() -> None:
            try:
                self._worker.execute(request)
                self._run_service.mark_completed(run_id)
            except Exception as exc:
                self._run_service.mark_failed(
                    run_id=run_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            finally:
                self._futures.pop(run_id, None)

        future = self._executor.submit(_run)
        self._futures[run_id] = future


    def shutdown(self, wait: bool = True) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait)


    def cancel_execution(self, run_id: str) -> None:
        future = self._futures.get(run_id)
        if future:
            future.cancel()

        if isinstance(self._worker, CancellableExecutionWorker):
            self._worker.cancel(run_id)


