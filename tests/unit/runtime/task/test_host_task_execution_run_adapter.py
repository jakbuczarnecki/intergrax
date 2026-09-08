# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import asyncio

import pytest
from echo.echo_agent import EchoAgent
from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.runs.default_service import DefaultRunService
from intergrax.fastapi_core.runs.models import CreateRunRequest, RunStatus
from intergrax.runtime.execution.nexus_host_execution import build_host_task_execution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.host_task_execution_run_adapter import HostTaskExecutionRunAdapter
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import DummyRunStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _echo_stack() -> tuple[DefaultRunService, HostTaskExecutionRunAdapter, DummyRunStore]:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    nexus_loop = NexusLoop(registry)
    host_execution = build_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    adapter = HostTaskExecutionRunAdapter(host_execution)
    store = DummyRunStore()
    service = DefaultRunService(store, adapter)
    adapter.bind_run_service(service)
    return service, adapter, store


@pytest.mark.asyncio
async def test_host_task_execution_run_adapter_routes_through_host_execution() -> None:
    service, adapter, store = _echo_stack()
    run = store.create()
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello runs",
        context=TaskContext(capability="echo.basic"),
    )
    request = ExecutionRequest(
        run_id=run.run_id,
        tenant_id="t1",
        user_id="u1",
        input_payload=task_to_execution_payload(task),
    )

    await adapter.start_execution(request)

    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert "hello runs" in final.result_payload["answer"]


def test_default_run_service_forwards_create_request_payload_to_execution() -> None:
    service, adapter, store = _echo_stack()
    context = RequestContext(
        request_id="req-j2",
        tenant_id="t1",
        user_id="u1",
        auth=None,
        path="/runs",
        method="POST",
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="via create_run",
        context=TaskContext(capability="echo.basic"),
    )
    background_tasks = BackgroundTasks()
    run = service.create_run(
        context,
        background_tasks,
        create_request=CreateRunRequest(payload=task_to_execution_payload(task)),
    )
    assert run.status == RunStatus.PENDING

    for bg_task in background_tasks.tasks:
        result = bg_task.func(*bg_task.args, **bg_task.kwargs)
        if asyncio.iscoroutine(result):
            asyncio.run(result)

    final = store.get(run.run_id)
    assert final.status == RunStatus.COMPLETED
    assert final.result_payload is not None
    assert "via create_run" in final.result_payload["answer"]
    assert service._execution_adapter.host_execution is adapter.host_execution
