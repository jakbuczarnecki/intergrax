from __future__ import annotations

import asyncio
import time
from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus


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


def test_execution_pipeline_threaded() -> None:
    store = DummyRunStore()
    worker = ExecutionWorker(store)
    adapter = ThreadedExecutionAdapter(worker)

    service = DefaultRunService(store=store, execution_adapter=adapter)

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

    # Wait briefly for background thread to finish
    for _ in range(20):
        if store.get(run.run_id).status == RunStatus.COMPLETED:
            break
        time.sleep(0.05)

    final = store.get(run.run_id)

    assert run.status == RunStatus.PENDING
    assert final.status == RunStatus.COMPLETED
