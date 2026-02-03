import asyncio
from typing import Optional
from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.inprocess_adapter import InProcessExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.models import RunResponse
from tests._support.builder import DummyRunStore



class ResultWorker(ExecutionWorker):
    def execute(self, request: ExecutionRequest) -> Optional[dict]:
        return {"ok": True, "run_id": request.run_id}


def test_inprocess_execution_persists_result_payload() -> None:
    store = DummyRunStore()
    service = DefaultRunService(store=store, execution_adapter=None)

    worker = ResultWorker()
    adapter = InProcessExecutionAdapter(worker=worker, run_service=service)
    service._execution_adapter = adapter

    ctx = RequestContext(
        request_id="req-1",
        tenant_id="t",
        user_id="u",
        auth=None,
        path="/runs",
        method="POST",
    )

    bg = BackgroundTasks()
    run = service.create_run(ctx, bg)

    for task in bg.tasks:
        result = task.func(*task.args, **task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload == {"ok": True, "run_id": run.run_id}
