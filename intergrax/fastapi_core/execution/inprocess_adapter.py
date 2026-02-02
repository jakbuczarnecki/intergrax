# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.execution.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker


class InProcessExecutionAdapter(ExecutionAdapter):
    def __init__(self, worker: ExecutionWorker) -> None:
        self._worker = worker

    async def start_execution(self, request: ExecutionRequest) -> None:
        # In-process, synchronous execution
        self._worker.execute(request)
