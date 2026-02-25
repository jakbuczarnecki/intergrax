# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
import asyncio

from intergrax.fastapi_core.execution.adapters.inprocess_adapter import InProcessExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.service import RunService

pytestmark = pytest.mark.unit



class FailingWorker:
    """Test double – minimal execution contract."""

    def execute(self, run_id: str) -> None:
        raise RuntimeError("boom")



class SpyRunService(RunService):
    """Spy to verify orchestration contract."""

    def __init__(self) -> None:
        self.failed_called = False
        self.failed_run_id: str | None = None
        self.error_type: str | None = None
        self.error_message: str | None = None
        self.completed_called = False

    def mark_completed(self, run_id: str) -> None:
        self.completed_called = True

    def mark_failed(
        self,
        run_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        self.failed_called = True
        self.failed_run_id = run_id
        self.error_type = error_type
        self.error_message = error_message



def test_execution_adapter_calls_mark_failed_on_worker_exception() -> None:
    worker = FailingWorker()
    service = SpyRunService()
    adapter = InProcessExecutionAdapter(worker=worker, run_service=service)

    request = ExecutionRequest(
        run_id="test-run",
        tenant_id="t",
        user_id="u",
        input_payload={},
        metadata={},
    )

    run_id = "test-run"

    asyncio.run(adapter.start_execution(request))

    assert service.failed_called is True
    assert service.failed_run_id == run_id
    assert "boom" in service.error_message
