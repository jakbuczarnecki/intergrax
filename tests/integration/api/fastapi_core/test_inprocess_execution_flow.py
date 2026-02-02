from __future__ import annotations
import asyncio

from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.inprocess_adapter import InProcessExecutionAdapter
from intergrax.fastapi_core.execution.simple_worker import SimpleExecutionWorker
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus


# Minimal in-memory store for this test
class DummyRunStore:
    def __init__(self) -> None:
        self._runs = {}

    def create(self):
        run_id = "r1"
        run = type("Run", (), {"run_id": run_id, "status": RunStatus.PENDING})
        self._runs[run_id] = run
        return run

    def get(self, run_id: str):
        return self._runs[run_id]

    def update_status(self, run_id: str, status: RunStatus):
        current = self._runs[run_id]
        updated = type("Run", (), {"run_id": current.run_id, "status": status})
        self._runs[run_id] = updated
        return updated


def test_execution_pipeline_inprocess() -> None:
    store = DummyRunStore()
    worker = SimpleExecutionWorker()
    service = DefaultRunService(store, execution_adapter=None)
    adapter = InProcessExecutionAdapter(worker, run_service=service)
    service._execution_adapter = adapter

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
    # Manually execute background tasks (FastAPI runtime simulation)
    for task in background_tasks.tasks:
        result = task.func(*task.args, **task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    # Execution happens synchronously in-process
    final = store.get(run.run_id)

    assert run.status == RunStatus.PENDING
    assert final.status == RunStatus.COMPLETED
