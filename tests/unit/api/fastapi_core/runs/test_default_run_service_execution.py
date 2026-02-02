import asyncio
from typing import Dict
from uuid import uuid4

from fastapi import BackgroundTasks
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus
from intergrax.fastapi_core.runs.service import RunService
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

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> RunResponse:
        current = self._runs[run_id]
        updated = RunResponse(
            run_id=current.run_id,
            status=status,
            error_type=error_type,
            error_message=error_message,
        )
        self._runs[run_id] = updated
        return updated

    

class NoOpExecutionAdapter(ExecutionAdapter):
    def __init__(self, run_service: RunService) -> None:
        self._run_service = run_service

    async def start_execution(self, request: ExecutionRequest) -> None:
        run_id = request.run_id
        self._run_service.mark_running(run_id)
        self._run_service.mark_completed(run_id)


def test_create_run_delegates_execution_and_keeps_pending() -> None:
    store = DummyRunStore()

    service = DefaultRunService(
        store=store,
        execution_adapter=None,  # temporary
    )

    adapter = NoOpExecutionAdapter(service)
    service._execution_adapter = adapter  # wiring

    context = RequestContext(
        request_id="req-1",
        tenant_id="tenant-1",
        user_id="user-1",
        auth=None,
        path="/runs",
        method="POST",
    )

    background_tasks = BackgroundTasks()

    run = service.create_run(context, background_tasks)

    # still PENDING immediately
    assert run.status == RunStatus.PENDING

    # simulate FastAPI background execution
    for task in background_tasks.tasks:
        result = task.func(*task.args, **task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    # now execution finished
    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED

