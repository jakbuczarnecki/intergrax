# © Artur Czarnecki. All rights reserved.

"""TLX-TL11 — MCP catalog discovery-only qualification (FastMCP protocol)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import intergrax.applications._shared.mcp_catalog_tools as mcp_catalog_tools_module

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastmcp import Client
from pydantic import BaseModel

from intergrax.applications._shared.fastapi_mcp import couple_fastapi_with_mcp
from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.mcp_result_bounds import emit_bounded_mcp_list
from fastmcp.exceptions import ToolError
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _noop_handler() -> ToolHandler[_In, _Out]:
    async def _run(_input: _In, _ctx: object) -> _Out:
        raise AssertionError("catalog discovery must not invoke tool handlers")

    return _run


def _contract(
    tool_id: str,
    *,
    description: str = "desc",
    description_short: str | None = None,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=description,
        description_short=description_short,
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="qual",
    )


def _build_mcp(
    registry: ToolRegistry | None,
    *,
    host: HostTaskExecutionPort | None = None,
) -> object:
    port = host or AsyncMock(spec=HostTaskExecutionPort)
    return build_nexus_mcp_server(
        name="tl11-discovery-qual",
        host_execution=port,
        registry=AgentRegistry(),
        default_capability="qual.cap",
        tool_registry=registry,
    )


async def _tool_names(mcp: object) -> set[str]:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    return {tool.name for tool in tools}


@pytest.mark.asyncio
async def test_discovery_empty_catalog() -> None:
    registry = ToolRegistry()
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("list_catalog_tools", {})
    assert result.data == []


@pytest.mark.asyncio
async def test_discovery_single_tool_schema() -> None:
    registry = ToolRegistry()
    registry.register(_contract("qual.one"), _noop_handler())
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        listed = await client.call_tool("list_catalog_tools", {})
        described = await client.call_tool("describe_catalog_tool", {"tool_id": "qual.one"})
    assert len(listed.data) == 1
    item = listed.data[0]
    assert item["name"] == "qual.one"
    assert "inputSchema" in item
    assert described.data["name"] == "qual.one"
    assert described.data["annotations"]["category"] == "qual"


@pytest.mark.asyncio
async def test_discovery_multiple_tools_sorted_export() -> None:
    registry = ToolRegistry()
    for tool_id in ("qual.b", "qual.a", "qual.c"):
        registry.register(_contract(tool_id), _noop_handler())
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("list_catalog_tools", {})
    names = {item["name"] for item in result.data}
    assert names == {"qual.a", "qual.b", "qual.c"}


@pytest.mark.asyncio
async def test_discovery_unknown_tool_bounded_error() -> None:
    registry = ToolRegistry()
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("describe_catalog_tool", {"tool_id": "missing.id"})
    assert result.data["error"] == "unknown_tool:missing.id"


@pytest.mark.asyncio
async def test_discovery_invalid_empty_tool_id() -> None:
    registry = ToolRegistry()
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("describe_catalog_tool", {"tool_id": ""})
    assert result.data["error"] == "unknown_tool:"


@pytest.mark.asyncio
async def test_discovery_oversized_catalog_raises() -> None:
    registry = ToolRegistry()
    for index in range(50):
        registry.register(
            _contract(f"qual.big.{index}", description="z" * 200),
            _noop_handler(),
        )
    mcp = _build_mcp(registry)

    def _tight_list(payload: list[object]) -> list[object]:
        return emit_bounded_mcp_list(payload, max_bytes=256)

    with patch.object(mcp_catalog_tools_module, "emit_bounded_mcp_list", _tight_list):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="mcp_payload_byte_limit_exceeded"):
                await client.call_tool("list_catalog_tools", {})


@pytest.mark.asyncio
async def test_discovery_oversized_description_returns_byte_limit_error() -> None:
    registry = ToolRegistry()
    registry.register(_contract("qual.huge", description="ż" * 2_000_000), _noop_handler())
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("describe_catalog_tool", {"tool_id": "qual.huge"})
    assert result.data["error"] == "mcp_payload_byte_limit_exceeded"
    assert result.data["tool_id"] == "qual.huge"


@pytest.mark.asyncio
async def test_discovery_unicode_serialization() -> None:
    registry = ToolRegistry()
    registry.register(
        _contract("qual.unicode", description="Opis — 日本語 🔧"),
        _noop_handler(),
    )
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        result = await client.call_tool("list_catalog_tools", {})
    assert "日本語" in result.data[0]["description"]


@pytest.mark.asyncio
async def test_discovery_concurrent_calls() -> None:
    registry = ToolRegistry()
    registry.register(_contract("qual.concurrent"), _noop_handler())
    mcp = _build_mcp(registry)

    async def _list_once() -> int:
        async with Client(mcp) as client:
            result = await client.call_tool("list_catalog_tools", {})
        return len(result.data)

    counts = await asyncio.gather(*(_list_once() for _ in range(8)))
    assert counts == [1] * 8


@pytest.mark.asyncio
async def test_discovery_mount_disabled_excludes_catalog_tools() -> None:
    mcp = _build_mcp(None)
    names = await _tool_names(mcp)
    assert "list_catalog_tools" not in names
    assert "describe_catalog_tool" not in names


@pytest.mark.asyncio
async def test_discovery_mount_enabled_exposes_catalog_tools() -> None:
    registry = ToolRegistry()
    mcp = _build_mcp(registry)
    names = await _tool_names(mcp)
    assert {"list_catalog_tools", "describe_catalog_tool"}.issubset(names)


@pytest.mark.asyncio
async def test_discovery_no_per_tool_catalog_invoke_surface() -> None:
    registry = ToolRegistry()
    registry.register(_contract("qual.invoke"), _noop_handler())
    mcp = _build_mcp(registry)
    names = await _tool_names(mcp)
    forbidden = {
        name
        for name in names
        if name not in {"list_agents", "run_agent", "list_catalog_tools", "describe_catalog_tool"}
    }
    assert forbidden == set()


@pytest.mark.asyncio
async def test_discovery_invoke_catalog_tool_absent_without_host_invoker() -> None:
    registry = ToolRegistry()
    mcp = _build_mcp(registry)
    names = await _tool_names(mcp)
    assert "invoke_catalog_tool" not in names


@pytest.mark.asyncio
async def test_discovery_does_not_mutate_registry() -> None:
    registry = ToolRegistry()
    registry.register(_contract("qual.stable"), _noop_handler())
    mcp = _build_mcp(registry)
    async with Client(mcp) as client:
        await client.call_tool("list_catalog_tools", {})
        await client.call_tool("describe_catalog_tool", {"tool_id": "qual.stable"})
    assert registry.tool_ids() == ["qual.stable"]


def test_mcp_http_ingress_rejects_missing_harness_credentials() -> None:
    api = FastAPI()
    api.state.harness_auth = HarnessAuthState(
        require_api_key=True,
        resolved_api_key="qual-secret",
        api_key_principal_tenant_id="bound-tenant",
    )
    mcp = _build_mcp(ToolRegistry())
    combined = couple_fastapi_with_mcp(api, mcp, mount_path="/mcp")
    with TestClient(combined) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_mcp_run_agent_lab_mode_accepts_client_tenant_without_harness_auth() -> None:
    """Lab/in-process MCP without harness auth wiring — client tenant is intake-only (not production)."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(
        return_value=TaskResult(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            state=TaskState.COMPLETED,
            answer="ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )
    )
    mcp = _build_mcp(ToolRegistry(), host=port)
    async with Client(mcp) as client:
        await client.call_tool(
            "run_agent",
            {
                "message": "x",
                "tenant_id": "lab-tenant",
                "user_id": "lab-user",
            },
        )
    task = port.execute.await_args.args[0]
    assert task.tenant_id == "lab-tenant"
    assert task.user_id == "lab-user"
