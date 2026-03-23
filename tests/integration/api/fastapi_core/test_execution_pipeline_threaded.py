from __future__ import annotations

import asyncio
import time
from fastapi import BackgroundTasks
import pytest

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.adapters.simple_worker import SimpleExecutionWorker
from intergrax.fastapi_core.execution.adapters.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from testing_support.builder import DummyRunStore

pytestmark = pytest.mark.integration


def test_execution_pipeline_threaded() -> None:
    store = DummyRunStore()

    service = DefaultRunService(store=store, execution_adapter=None)

    worker = SimpleExecutionWorker()
    adapter = ThreadedExecutionAdapter(worker, run_service=service)

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

    # Wait briefly for background thread to finish
    for _ in range(20):
        if store.get(run.run_id).status == RunStatus.COMPLETED:
            break
        time.sleep(0.05)

    final = store.get(run.run_id)

    assert run.status == RunStatus.PENDING
    assert final.status == RunStatus.COMPLETED
