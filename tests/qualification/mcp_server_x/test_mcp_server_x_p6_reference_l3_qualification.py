# © Artur Czarnecki. All rights reserved.

"""MCP-P6-02 / MCP-ACC-01 — reference qualification server on streamable HTTP L3."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
import uvicorn
from fastapi import FastAPI
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution
from tests.qualification.mcp_server_x.reference_app.build import (
    REFERENCE_TENANT,
    build_mcp_server_x_reference_qualification_server,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.network, pytest.mark.qualification]


@contextmanager
def _uvicorn_base_url(app: FastAPI) -> Iterator[str]:
    bind = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bind.bind(("127.0.0.1", 0))
    port = bind.getsockname()[1]
    bind.close()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 12.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("uvicorn_mcp_qualification_startup_timeout")
    try:
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            thread.join(timeout=10.0)


def _harness_host_with_reference_mcp() -> FastAPI:
    registry = AgentRegistry()
    nexus = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus, orchestration_triggers=frozenset())
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=host_execution,
        registry=registry,
    )
    api = FastAPI()
    return couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")


async def _streamable_list_tools(base_url: str) -> list[str]:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(base_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [item.name for item in tools.tools]


async def _streamable_call_qual_echo(base_url: str, message: str) -> dict[str, object]:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(base_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("qual_echo", {"message": message})
            if result.content and result.content[0].type == "text":
                import json

                return json.loads(result.content[0].text)
            return {}


@pytest.mark.asyncio
async def test_p6_reference_streamable_http_discovery_and_custom_tool() -> None:
    app = _harness_host_with_reference_mcp()
    with _uvicorn_base_url(app) as base_url:
        names = await _streamable_list_tools(base_url)
        assert "qual_echo" in names
        assert "list_agents" in names
        payload = await _streamable_call_qual_echo(base_url, "l3-ok")
        assert payload.get("echo") == "l3-ok"


@pytest.mark.asyncio
async def test_p6_reference_streamable_http_run_agent_authorized_path() -> None:
    app = _harness_host_with_reference_mcp()
    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id="task-p6",
            run_id="run-p6",
            state=TaskState.COMPLETED,
            answer="done",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        ),
    ):
        with _uvicorn_base_url(app) as base_url:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(base_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "run_agent",
                        {
                            "message": "hi",
                            "capability": "mcp.p6.reference",
                            "tenant_id": REFERENCE_TENANT,
                            "user_id": "p6-user",
                        },
                    )
                    assert result.isError is not True
