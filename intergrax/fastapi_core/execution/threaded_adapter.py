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
    ) -> None:
        self._worker = worker
        self._executor = executor or ThreadPoolExecutor(max_workers=4)

    async def start_execution(self, request: ExecutionRequest) -> None:
        # Dispatch execution to a background thread
        self._executor.submit(self._worker.execute, request)
