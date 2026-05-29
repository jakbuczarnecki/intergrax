# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest
from echo.echo_agent import EchoAgent
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from intergrax.fastapi_core.app_factory import create_app
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunStatus
from tests.unit.api.fastapi_core.budget.test_budget_required import AllowAllAuthProvider
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.nexus_task_execution_adapter import NexusTaskExecutionAdapter
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def _build_run_service() -> DefaultRunService:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    task_runner = UnifiedTaskRunner(NexusLoop(registry))
    adapter = NexusTaskExecutionAdapter(task_runner)
    store = InMemoryRunStore()
    service = DefaultRunService(store, adapter)
    adapter.bind_run_service(service)
    return service


def test_run_service_and_http_chat_share_unified_task_runner_path() -> None:
    service = _build_run_service()
    task_runner = service._execution_adapter.task_runner

    context = RequestContext(
        request_id="req-j2-int",
        tenant_id="t1",
        user_id="u1",
        auth=None,
        path="/runs",
        method="POST",
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="shared runner path",
        context=TaskContext(capability="echo.basic"),
    )
    background_tasks = BackgroundTasks()
    run = service.create_run(
        context,
        background_tasks,
        create_request=CreateRunRequest(payload=task_to_execution_payload(task)),
    )

    for bg_task in background_tasks.tasks:
        result = bg_task.func(*bg_task.args, **bg_task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    final = service.get_run(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert "shared runner path" in final.result_payload["answer"]
    assert service._execution_adapter.task_runner is task_runner


def test_fastapi_runs_endpoint_executes_task_payload_via_unified_runner() -> None:
    service = _build_run_service()
    app = create_app(
        ApiConfig(
            environment=ApiEnvironment.DEV,
            run_service=service,
            run_store=service._store,
            auth_provider=AllowAllAuthProvider(),
        )
    )
    client = TestClient(app)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="http runs payload",
        context=TaskContext(capability="echo.basic"),
    )
    response = client.post(
        "/runs",
        json={"payload": task_to_execution_payload(task)},
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["run_id"]
    final = service.get_run(run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert "http runs payload" in final.result_payload["answer"]
