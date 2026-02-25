from __future__ import annotations

import asyncio
import pytest

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.adapters.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.execution.worker_contract import CancellableExecutionWorker
from intergrax.fastapi_core.runs.service import RunService

pytestmark = pytest.mark.integration

class DummyWorker(CancellableExecutionWorker):
    def execute(self, request: ExecutionRequest) -> None:
        return
    
    def cancel(self, run_id: str) -> None:
        return


class NoOpRunService(RunService):
    def mark_running(self, run_id: str) -> None:
        pass

    def mark_completed(self, run_id: str) -> None:
        pass

    def mark_failed(self, run_id: str, error_type: str, error_message: str) -> None:
        pass


def test_threaded_execution_adapter_shutdown_blocks_new_execution() -> None:
    adapter = ThreadedExecutionAdapter(
        worker=DummyWorker(),
        run_service=NoOpRunService(),
        max_workers=1,
    )

    adapter.shutdown(wait=True)

    request = ExecutionRequest(
        run_id="r1",
        tenant_id="t1",
        user_id=None,
        input_payload={},
    )

    with pytest.raises(RuntimeError):
        asyncio.run(adapter.start_execution(request))
