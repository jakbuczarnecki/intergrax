# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    exposed_builtin_tool_names,
    nexus_legacy_server_definition,
)
from intergrax.contracts.mcp_server_definition import (
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    build_mcp_server_definition,
    resolve_effective_bindings,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from pydantic import BaseModel
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

pytestmark = [pytest.mark.unit]


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _register_tool(registry: ToolRegistry, tool_id: str) -> None:
    contract = ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="test tool",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="test",
    )

    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out()

    registry.register(contract, _run)


def test_compose_minimal_host_tools() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())

    definition = build_mcp_server_definition(
        server_id="test-srv",
        display_name="Test MCP",
        default_capability="test.cap",
        bindings=(
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
        ),
    )
    deps = McpServerCompositionDeps(host_execution=host_execution, registry=registry)
    compose_mcp_server_from_definition(definition, deps)
    assert "list_agents" in exposed_builtin_tool_names(definition)
    assert "run_agent" in exposed_builtin_tool_names(definition)


@pytest.mark.asyncio
async def test_build_nexus_facade_still_exposes_core_tools() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    mcp = build_nexus_mcp_server(
        name="Facade",
        host_execution=host_execution,
        registry=registry,
        default_capability="test.cap",
    )
    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=TaskResult(
            task_id="task-1",
            run_id="run-1",
            state=TaskState.COMPLETED,
            answer="ok",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        ),
    ):
        from fastmcp import Client

        async with Client(mcp) as client:
            tools = await client.list_tools()
            names = {item.name for item in tools}
            assert "list_agents" in names
            assert "run_agent" in names


def test_catalog_exposure_allowlist_limits_bindings() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="list_catalog_tools",
                builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
            ),
            McpCatalogToolBinding(binding_id="keep", tool_id="tool.keep"),
            McpCatalogToolBinding(binding_id="drop", tool_id="tool.drop"),
        ),
        exposure_policy=McpExposurePolicy(
            enabled=True,
            allowed_binding_ids=(
                "list_catalog_tools",
                "keep",
            ),
        ),
    )
    exposed = resolve_effective_bindings(definition)
    tool_ids = {b.tool_id for b in exposed if isinstance(b, McpCatalogToolBinding)}
    assert tool_ids == {"tool.keep"}


def test_nexus_legacy_definition_includes_catalog_surface_when_registry_present() -> None:
    definition = nexus_legacy_server_definition(
        server_id="id",
        display_name="Name",
        default_capability="cap",
        default_tenant_id="t",
        default_user_id="u",
        catalog_tool_ids=["tool.a"],
        include_catalog_surface=True,
    )
    builtins = {b.builtin for b in definition.bindings if isinstance(b, McpHostBuiltinBinding)}
    assert McpHostBuiltinId.INVOKE_CATALOG_TOOL in builtins
    assert any(isinstance(b, McpCatalogToolBinding) and b.tool_id == "tool.a" for b in definition.bindings)


def test_compose_with_catalog_registry_filters_allowlist() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "tool.keep")
    _register_tool(tool_registry, "tool.drop")

    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="list_catalog_tools",
                builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
            ),
            McpHostBuiltinBinding(
                binding_id="describe_catalog_tool",
                builtin=McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
            ),
            McpCatalogToolBinding(binding_id="keep", tool_id="tool.keep"),
            McpCatalogToolBinding(binding_id="drop", tool_id="tool.drop"),
        ),
        exposure_policy=McpExposurePolicy(
            enabled=True,
            allowed_binding_ids=("list_catalog_tools", "describe_catalog_tool", "keep"),
        ),
    )
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        tool_registry=tool_registry,
    )
    mcp = compose_mcp_server_from_definition(definition, deps)
    from fastmcp import Client

    async def _list() -> set[str]:
        async with Client(mcp) as client:
            result = await client.call_tool("list_catalog_tools", {})
            names = {item["name"] for item in result.data}
            return names

    names = asyncio_run(_list())
    assert names == {"tool.keep"}


def asyncio_run(coro):  # type: ignore[no-untyped-def]
    import asyncio

    return asyncio.run(coro)


@pytest.mark.asyncio
async def test_host_custom_tool_compose_discover_and_invoke() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())
    calls: list[tuple[str, str]] = []

    async def alpha_tool(message: str, marker: str = "alpha") -> dict[str, object]:
        calls.append((message, marker))
        return {"tool": "alpha", "message": message, "marker": marker}

    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(McpHostCustomBinding(binding_id="alpha", tool_name="alpha_tool"),),
    )
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        custom_host_tools={"alpha_tool": alpha_tool},
    )
    mcp = compose_mcp_server_from_definition(definition, deps)
    from fastmcp import Client

    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {item.name for item in tools}
        assert names == {"alpha_tool"}
        schema = next(item for item in tools if item.name == "alpha_tool").inputSchema
        assert set(schema.get("properties", {})) == {"message", "marker"}
        assert "_handler" not in schema.get("properties", {})
        result = await client.call_tool("alpha_tool", {"message": "hello"})
        assert result.data == {"tool": "alpha", "message": "hello", "marker": "alpha"}
    assert calls == [("hello", "alpha")]


@pytest.mark.asyncio
async def test_multiple_host_custom_tools_invoke_distinct_handlers() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())

    async def alpha_tool(message: str) -> dict[str, object]:
        return {"which": "alpha", "message": message}

    async def beta_tool(message: str) -> dict[str, object]:
        return {"which": "beta", "message": message}

    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostCustomBinding(binding_id="alpha", tool_name="alpha_tool"),
            McpHostCustomBinding(binding_id="beta", tool_name="beta_tool"),
        ),
    )
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        custom_host_tools={"alpha_tool": alpha_tool, "beta_tool": beta_tool},
    )
    mcp = compose_mcp_server_from_definition(definition, deps)
    from fastmcp import Client

    async with Client(mcp) as client:
        alpha = await client.call_tool("alpha_tool", {"message": "a"})
        beta = await client.call_tool("beta_tool", {"message": "b"})
    assert alpha.data == {"which": "alpha", "message": "a"}
    assert beta.data == {"which": "beta", "message": "b"}


@pytest.mark.asyncio
async def test_host_custom_tool_rejects_kwargs_handlers() -> None:
    registry = AgentRegistry()
    nexus_loop = NexusLoop(registry)
    host_execution = build_harness_host_task_execution(nexus_loop, orchestration_triggers=frozenset())

    async def bad_tool(**kwargs: object) -> dict[str, object]:
        return {"kwargs": kwargs}

    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(McpHostCustomBinding(binding_id="bad", tool_name="bad_tool"),),
    )
    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        custom_host_tools={"bad_tool": bad_tool},
    )
    with pytest.raises(ValueError, match="explicit parameters"):
        compose_mcp_server_from_definition(definition, deps)
