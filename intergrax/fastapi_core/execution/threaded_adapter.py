# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from intergrax.fastapi_core.execution.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker


class ThreadedExecutionAdapter(ExecutionAdapter):
    def __init__(
        self,
        worker: ExecutionWorker,
        executor: Optional[ThreadPoolExecutor] = None,
        max_workers: int = 4,
    ) -> None:
        self._worker = worker

        if executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._owns_executor = True
        else:
            self._executor = executor
            self._owns_executor = False

    async def start_execution(self, request: ExecutionRequest) -> None:
        self._executor.submit(self._worker.execute, request)

    def shutdown(self, wait: bool = True) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait)

