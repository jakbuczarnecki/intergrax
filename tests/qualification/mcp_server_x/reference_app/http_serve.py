# © Artur Czarnecki. All rights reserved.

"""Physical streamable HTTP helpers for the MCP-SERVER-X reference server (MCP-P7-01, L3 support).

Everything here serves a real ``uvicorn`` socket on ``127.0.0.1`` with an OS-assigned port
(``port=0``). Tests that use these helpers are L3 (physical streamable HTTP on loopback);
in-process ``fastmcp.Client(mcp)`` tests are L2 and must not use this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP
from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

MCP_MOUNT_PATH = "/mcp"
DEFAULT_CLIENT_TIMEOUT_S = 20.0


@dataclass(frozen=True, slots=True)
class ReferenceHostParts:
    """Registry + canonical host execution used behind a reference MCP server."""

    registry: AgentRegistry
    host_execution: HostTaskExecutionPort


def build_reference_host_parts(registry: AgentRegistry | None = None) -> ReferenceHostParts:
    """Canonical harness host task execution over a fresh (or shared) ``AgentRegistry``."""
    reg = registry if registry is not None else AgentRegistry()
    nexus = NexusLoop(reg)
    host_execution = build_harness_host_task_execution(nexus, orchestration_triggers=frozenset())
    return ReferenceHostParts(registry=reg, host_execution=host_execution)


def couple_reference_host(
    mcp: FastMCP,
    *,
    auth: HarnessAuthState | None = None,
    mount_path: str = MCP_MOUNT_PATH,
) -> FastAPI:
    """Mount *mcp* on a wrapper app exactly like production hosts (``couple_fastapi_with_mcp``).

    ``auth`` is placed on the inner FastAPI ``state.harness_auth`` (the production wiring); when it
    requires an API key the wrapper app gets the harness auth middleware.
    """
    api = FastAPI()
    if auth is not None:
        api.state.harness_auth = auth
    return couple_fastapi_with_mcp(api, mcp, mount_path=mount_path)


@dataclass(frozen=True, slots=True)
class ServedMcp:
    """A running uvicorn instance: endpoint URL, bound port and the raw ASGI app."""

    url: str
    port: int
    app: FastAPI


@contextmanager
def served_mcp(app: FastAPI, *, mount_path: str = MCP_MOUNT_PATH) -> Iterator[ServedMcp]:
    """Serve *app* on ``127.0.0.1`` with an OS-assigned port (``port=0``) until the block exits."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        raise RuntimeError("uvicorn_mcp_reference_startup_timeout")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield ServedMcp(url=f"http://127.0.0.1:{port}{mount_path}/", port=port, app=app)
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            thread.join(timeout=10.0)


@asynccontextmanager
async def streamable_session(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Open a streamable HTTP MCP client session; yields ``(session, initialize_result)``."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            yield session, init


async def with_session(
    url: str,
    action: Any,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = DEFAULT_CLIENT_TIMEOUT_S,
) -> Any:
    """Run ``await action(session, init_result)`` in one bounded streamable HTTP session."""

    async def _run() -> Any:
        async with streamable_session(url, headers=headers) as (session, init):
            return await action(session, init)

    return await asyncio.wait_for(_run(), timeout=timeout_s)


async def list_tool_names(url: str, *, headers: dict[str, str] | None = None) -> list[str]:
    async def _action(session: Any, _init: Any) -> list[str]:
        result = await session.list_tools()
        return [item.name for item in result.tools]

    return await with_session(url, _action, headers=headers)


async def call_tool(
    url: str,
    name: str,
    arguments: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> Any:
    async def _action(session: Any, _init: Any) -> Any:
        return await session.call_tool(name, arguments or {})

    return await with_session(url, _action, headers=headers)


def tool_payload(result: Any) -> dict[str, Any]:
    """Structured payload of a ``CallToolResult`` (structured content, else JSON text content)."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            with contextlib.suppress(ValueError):
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
    return {}


def error_text(result: Any) -> str | None:
    """Text of an MCP error result (``isError``), else ``None``."""
    if not getattr(result, "isError", False):
        return None
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", None) if content else None
    return text if isinstance(text, str) else ""


def jsonrpc_post(
    url: str,
    method: str,
    params: dict[str, object] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout_s: float = 10.0,
) -> httpx.Response:
    """Raw streamable HTTP POST (no session) for HTTP-level assertions such as ``401``."""
    merged = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    merged.update(headers or {})
    body: dict[str, object] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return httpx.post(url, json=body, headers=merged, timeout=timeout_s)


def completed_task_result(*, task_id: str = "task-p7", run_id: str = "run-p7") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


@contextmanager
def patched_task_delegate(result: TaskResult | None = None) -> Iterator[AsyncMock]:
    """Replace the bound agentic delegate so ``run_agent`` reaches a terminal state without an LLM."""
    mock = AsyncMock(return_value=result or completed_task_result())
    with patch("intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute", mock):
        yield mock
