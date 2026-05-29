# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from echo.echo_agent import EchoAgent
from intergrax.debug.app import create_debug_app
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState

pytestmark = [pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_debug_api_reads_injected_trace_store_without_sqlite() -> None:
    trace_store = InMemoryRunTraceStore()
    registry = AgentRegistry()
    registry.register(EchoAgent())
    nexus_loop = NexusLoop(registry, trace_store=trace_store)
    app = create_debug_app(
        registry=registry,
        nexus_loop=nexus_loop,
        trace_store=trace_store,
    )

    result = await nexus_loop.handle_task(
        Task(
            tenant_id="lab",
            user_id="tester",
            message="in-memory trace proof",
            context=TaskContext(capability="echo.basic"),
        )
    )
    assert result.state == TaskState.COMPLETED

    with TestClient(app) as client:
        detail = client.get(f"/debug/tasks/{result.task_id}", params={"tenant": "lab"})
        assert detail.status_code == 200, detail.text
        body = detail.json()
        assert body["run_id"] == result.task_id
        assert body["event_count"] > 0

        trace = client.get(
            f"/debug/tasks/{result.task_id}/trace",
            params={"tenant": "lab"},
        )
        assert trace.status_code == 200, trace.text
        assert len(trace.json()["trace_events"]) > 0

        listed = client.get("/debug/tasks", params={"tenant": "lab", "limit": 10})
        assert listed.status_code == 200
        run_ids = {row["run_id"] for row in listed.json()["runs"]}
        assert result.task_id in run_ids
