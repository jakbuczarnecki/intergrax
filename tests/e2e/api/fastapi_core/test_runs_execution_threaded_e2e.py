from __future__ import annotations

import time
from fastapi.testclient import TestClient

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.execution.simple_worker import SimpleExecutionWorker
from intergrax.fastapi_core.execution.threaded_adapter import ThreadedExecutionAdapter
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import RunStatus
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore
from tests.unit.api.fastapi_core.budget.test_budget_required import AllowAllAuthProvider


def test_runs_execution_threaded_e2e() -> None:
    store = InMemoryRunStore()
    service = DefaultRunService(store, execution_adapter=None)

    worker = SimpleExecutionWorker()
    adapter = ThreadedExecutionAdapter(worker, run_service=service)

    service._execution_adapter = adapter

    config = ApiConfig(
        environment=ApiEnvironment.DEV,
        run_store=store,
        execution_adapter=adapter,
        auth_provider=AllowAllAuthProvider(),
    )

    app = create_app(config)
    client = TestClient(app)

    # Create run
    resp = client.post(
        "/runs",
        json={
            "payload": {},
        },)
    assert resp.status_code == 201

    run = resp.json()
    run_id = run["run_id"]

    assert run["status"] == RunStatus.PENDING.value

    # Poll until completed
    for _ in range(30):
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        if status == RunStatus.COMPLETED.value:
            break
        time.sleep(0.05)

    assert status == RunStatus.COMPLETED.value
