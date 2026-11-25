# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.debug.app import create_debug_app
from intergrax.runtime.execution.harness_task_execution_port import (
    build_harness_root_task_execution_port,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.admitted_root_governance_identity import (
    lab_admitted_root_governance_identity_for_task,
)

from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_debug_api_reads_injected_trace_store_without_sqlite() -> None:
    trace_store = InMemoryRunTraceStore()
    registry = AgentRegistry()
    registry.register(EchoAgent())
    nexus_loop = NexusLoop(registry, trace_store=trace_store, production_mode=False)
    app = create_debug_app(
        registry=registry,
        nexus_loop=nexus_loop,
        trace_store=trace_store,
    )

    runner = UnifiedTaskRunner(
        build_harness_root_task_execution_port(nexus_loop),
        admitted_governance_identity_for_task=lab_admitted_root_governance_identity_for_task,
    )
    run_id = mint_run_id()
    result = await runner.run_task(
        Task(
            tenant_id="lab",
            user_id="tester",
            message="in-memory trace proof",
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id,
    )
    assert result.state == TaskState.COMPLETED
    assert result.run_id is not None
    assert result.run_id == run_id

    with TestClient(app) as client:
        detail = client.get(f"/debug/tasks/{result.run_id}", params={"tenant": "lab"})
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["run_id"] == result.run_id
        assert body["event_count"] > 0

        trace = client.get(
            f"/debug/tasks/{result.run_id}/trace",
            params={"tenant": "lab"},
        )
        assert trace.status_code == 200, trace.text
        assert len(trace.json()["trace_events"]) > 0

        listed = client.get("/debug/tasks", params={"tenant": "lab", "limit": 10})
        assert listed.status_code == 200
        run_ids = {row["run_id"] for row in listed.json()["runs"]}
        assert result.run_id in run_ids
