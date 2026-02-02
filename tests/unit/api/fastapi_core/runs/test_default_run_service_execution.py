from typing import Dict
from uuid import uuid4

from fastapi import BackgroundTasks
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore

class DummyRunStore(RunStore):
    def __init__(self) -> None:
        self._runs: Dict[str, RunResponse] = {}

    def create(self) -> RunResponse:
        run_id = uuid4().hex
        run = RunResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
        )
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> RunResponse:
        return self._runs[run_id]
    
    def cancel(self, run_id: str) -> RunResponse:
        raise AssertionError("Should not reach store.cancel() if transition invalid")

    def update_status(self, run_id: str, status: RunStatus) -> RunResponse:
        current = self._runs[run_id]
        updated = RunResponse(
            run_id=current.run_id,
            status=status,
        )
        self._runs[run_id] = updated
        return updated
    

class NoOpExecutionAdapter:
    async def start_execution(self, request: ExecutionRequest) -> None:
        return


def test_create_run_delegates_execution_and_keeps_pending() -> None:
    store = DummyRunStore()

    context = RequestContext(
        request_id="req-1",
        tenant_id="tenant-1",
        user_id="user-1",
        auth=None,
        path="/runs",
        method="POST",
    )

    background_tasks = BackgroundTasks()

    service = DefaultRunService(
        store=store,
        execution_adapter=NoOpExecutionAdapter(),
    )

    run = service.create_run(context, background_tasks)

    assert run.status == RunStatus.PENDING
