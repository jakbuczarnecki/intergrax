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
from testing_support.nexus_host_task_execution import (
    build_certified_internal_test_host_task_execution,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.host_task_execution_run_adapter import HostTaskExecutionRunAdapter
from intergrax.runtime.task.task_run_bridge import task_to_execution_payload
from testing_support.builder import DummyRunStore, build_task_for_tests, canonical_governed_execution_scope

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _echo_stack() -> tuple[DefaultRunService, HostTaskExecutionRunAdapter, DummyRunStore]:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    nexus_loop = NexusLoop(registry)
    host_execution = build_certified_internal_test_host_task_execution(nexus_loop)
    adapter = HostTaskExecutionRunAdapter(host_execution)
    store = DummyRunStore()
    service = DefaultRunService(store, adapter)
    adapter.bind_run_service(service)
    return service, adapter, store


@pytest.mark.asyncio
async def test_host_task_execution_run_adapter_routes_through_host_execution() -> None:
    service, adapter, store = _echo_stack()
    run = store.create()
    seed = "host-adapter-run"
    task = build_task_for_tests(
        seed=seed,
        tenant_id="t1",
        user_id="u1",
        message="hello runs",
    )
    request = ExecutionRequest(
        run_id=run.run_id,
        tenant_id="t1",
        user_id="u1",
        input_payload=task_to_execution_payload(task),
    )

    with canonical_governed_execution_scope(seed):
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
    seed = "host-adapter-create"
    task = build_task_for_tests(
        seed=seed,
        tenant_id="t1",
        user_id="u1",
        message="via create_run",
    )
    background_tasks = BackgroundTasks()
    with canonical_governed_execution_scope(seed):
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
