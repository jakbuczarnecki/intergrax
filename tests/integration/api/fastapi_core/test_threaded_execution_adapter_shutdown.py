from __future__ import annotations

import asyncio
import pytest

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.threaded_adapter import ThreadedExecutionAdapter


class DummyWorker:
    def execute(self, request: ExecutionRequest) -> None:
        return


def test_threaded_execution_adapter_shutdown_blocks_new_execution() -> None:
    adapter = ThreadedExecutionAdapter(worker=DummyWorker(), max_workers=1)

    adapter.shutdown(wait=True)

    request = ExecutionRequest(
        run_id="r1",
        tenant_id="t1",
        user_id=None,
        input_payload={},
    )

    with pytest.raises(RuntimeError):
        asyncio.run(adapter.start_execution(request))
