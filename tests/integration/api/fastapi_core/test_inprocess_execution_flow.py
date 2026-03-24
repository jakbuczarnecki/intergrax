from __future__ import annotations
import asyncio

from fastapi import BackgroundTasks
import pytest

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.adapters.inprocess_adapter import InProcessExecutionAdapter
from intergrax.fastapi_core.execution.adapters.simple_worker import SimpleExecutionWorker
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from testing_support.builder import DummyRunStore


pytestmark = pytest.mark.integration

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

    assert final.started_at is not None
    assert final.finished_at is not None
    assert final.duration_ms is not None
    assert final.duration_ms >= 0
