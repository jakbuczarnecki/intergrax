# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 lane L2: composition, exposure policy and retire semantics.

Level: **L2** (in-process FastMCP ``Client`` over the in-memory transport). This module is
**not a physical transport test**: no socket, no HTTP, no uvicorn. It is never counted as L3.

Scenarios (ticket ``T-IGX-035-H``, no product-path change):

* MCP-ACC-03 composition on the P6 reference server (exact tool set, bounded behaviour,
  fail-closed composition errors).
* MCP-ACC-04 exposure policy allowlist on composed host builtins, checked with ``list_tools``.
* MCP-ACC-16 (L2 part) a retired binding composes, is absent from ``list_tools`` and
  ``call_tool`` fails closed with no redirect to another binding.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from intergrax.applications._shared.mcp_developer_api import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
)
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.contracts.mcp_server_definition import (
    McpBindingVisibility,
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpServerDefinitionErrorCode,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    resolve_binding_for_invocation,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
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
from tests.qualification.mcp_server_x.reference_app.build import (
    REFERENCE_CAPABILITY,
    REFERENCE_SERVER_ID,
    REFERENCE_TENANT,
    build_mcp_server_x_reference_qualification_server,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REFERENCE_CORE_TOOLS = {"list_agents", "run_agent", "qual_echo"}
_REFERENCE_CATALOG_SURFACE = {"list_catalog_tools", "describe_catalog_tool", "invoke_catalog_tool"}


class _In(BaseModel):
    query: str = "x"


class _Out(BaseModel):
    value: str = "ok"


def _register_tool(registry: ToolRegistry, tool_id: str) -> None:
    contract = ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="p7 l2 composition tool",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="p7-l2",
    )

    async def _run(_input: _In, _ctx: object) -> _Out:
        return _Out()

    registry.register(contract, _run)


def _completed_result() -> TaskResult:
    return TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="p7-l2-done",
        authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
    )


def _real_harness_host(registry: AgentRegistry):  # type: ignore[no-untyped-def]
    return build_harness_host_task_execution(NexusLoop(registry), orchestration_triggers=frozenset())


def _recording_host() -> AsyncMock:
    """Host port double that records ``execute`` calls (used to prove "never started")."""
    port = AsyncMock(spec=HostTaskExecutionPort)
    port.execute = AsyncMock(return_value=_completed_result())
    return port


def _deps(
    *,
    host: object | None = None,
    registry: AgentRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    custom: dict | None = None,
) -> McpServerCompositionDeps:
    agent_registry = registry or AgentRegistry()
    return McpServerCompositionDeps(
        host_execution=host or _recording_host(),  # type: ignore[arg-type]
        registry=agent_registry,
        tool_registry=tool_registry,
        custom_host_tools=custom,
    )


async def _tool_names(mcp: object) -> set[str]:
    async with Client(mcp) as client:  # type: ignore[arg-type]
        return {item.name for item in await client.list_tools()}


# --------------------------------------------------------------------------------------
# MCP-ACC-03: composition on the reference server
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acc_03_reference_server_composes_exact_core_tool_set() -> None:
    registry = AgentRegistry()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_real_harness_host(registry),
        registry=registry,
    )
    assert await _tool_names(mcp) == _REFERENCE_CORE_TOOLS


@pytest.mark.asyncio
async def test_acc_03_reference_server_with_tool_registry_adds_catalog_surface() -> None:
    registry = AgentRegistry()
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "p7.l2.tool")
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_real_harness_host(registry),
        registry=registry,
        tool_registry=tool_registry,
    )
    # Catalog surface appears only with a registry; invoke_catalog_tool additionally needs a
    # host-injected invoker, so it is absent here (fail-closed: no invoke without invoker).
    assert await _tool_names(mcp) == _REFERENCE_CORE_TOOLS | {"list_catalog_tools", "describe_catalog_tool"}


@pytest.mark.asyncio
async def test_acc_03_reference_server_custom_tool_and_list_agents_work_in_process() -> None:
    registry = AgentRegistry()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_real_harness_host(registry),
        registry=registry,
    )
    async with Client(mcp) as client:
        echoed = await client.call_tool("qual_echo", {"message": "p7-l2"})
        agents = await client.call_tool("list_agents", {})
    assert echoed.data == {"echo": "p7-l2", "server_id": REFERENCE_SERVER_ID}
    assert agents.data == []


@pytest.mark.asyncio
async def test_acc_03_reference_server_run_agent_uses_canonical_host_execution() -> None:
    registry = AgentRegistry()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_real_harness_host(registry),
        registry=registry,
    )
    delegate_result = _completed_result()
    with patch(
        "intergrax.runtime.execution.host_task.TaskBoundAgenticDelegate.execute",
        new_callable=AsyncMock,
        return_value=delegate_result,
    ) as delegate:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "run_agent",
                {"message": "hello", "capability": REFERENCE_CAPABILITY},
            )
    delegate.assert_awaited_once()
    assert result.data["state"] == TaskState.COMPLETED.value
    assert result.data["answer"] == "p7-l2-done"
    request = delegate.await_args.args[0]
    assert request.input.message == "hello"
    assert request.input.capability == REFERENCE_CAPABILITY


@pytest.mark.asyncio
async def test_acc_03_reference_server_run_agent_lab_identity_uses_definition_defaults() -> None:
    host = _recording_host()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=host,
        registry=AgentRegistry(),
    )
    async with Client(mcp) as client:
        await client.call_tool("run_agent", {"message": "hello"})
    task = host.execute.await_args.args[0]
    # In-process client, no harness auth: definition defaults apply (not a bound principal).
    assert task.tenant_id == REFERENCE_TENANT
    assert task.context.capability == REFERENCE_CAPABILITY


@pytest.mark.asyncio
async def test_acc_03_reference_server_input_schemas_are_derived_from_handlers() -> None:
    registry = AgentRegistry()
    mcp = build_mcp_server_x_reference_qualification_server(
        host_execution=_real_harness_host(registry),
        registry=registry,
    )
    async with Client(mcp) as client:
        tools = {item.name: item for item in await client.list_tools()}
    run_schema = tools["run_agent"].inputSchema
    assert set(run_schema["properties"]) == {
        "message",
        "capability",
        "tenant_id",
        "user_id",
        "session_id",
        "intent",
    }
    assert run_schema["required"] == ["message"]
    echo_schema = tools["qual_echo"].inputSchema
    assert set(echo_schema["properties"]) == {"message"}
    assert "_handler" not in echo_schema["properties"]


@pytest.mark.asyncio
async def test_acc_03_missing_custom_handler_fails_closed_at_composition() -> None:
    definition = build_mcp_server_definition(
        server_id="p7-l2-missing-handler",
        display_name="P7 L2 missing handler",
        default_capability="p7.l2",
        bindings=(McpHostCustomBinding(binding_id="ghost", tool_name="ghost_tool"),),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        compose_mcp_server_from_definition(definition, _deps(custom={}))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_DEF_INVALID
    assert "ghost_tool" in str(exc.value)


@pytest.mark.asyncio
async def test_acc_03_var_kwargs_custom_handler_rejected_at_composition() -> None:
    async def bad_tool(**kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    definition = build_mcp_server_definition(
        server_id="p7-l2-kwargs",
        display_name="P7 L2 kwargs",
        default_capability="p7.l2",
        bindings=(McpHostCustomBinding(binding_id="bad", tool_name="bad_tool"),),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        compose_mcp_server_from_definition(definition, _deps(custom={"bad_tool": bad_tool}))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_DEF_INVALID


def test_acc_03_unknown_catalog_tool_binding_rejected_at_composition() -> None:
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "p7.l2.known")
    definition = build_mcp_server_definition(
        server_id="p7-l2-unknown-tool",
        display_name="P7 L2 unknown tool",
        default_capability="p7.l2",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="list_catalog_tools",
                builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
            ),
            McpCatalogToolBinding(binding_id="ghost", tool_id="p7.l2.ghost"),
        ),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        compose_mcp_server_from_definition(definition, _deps(tool_registry=tool_registry))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN_TOOL


def test_acc_03_catalog_binding_without_introspection_surface_rejected() -> None:
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "p7.l2.known")
    definition = build_mcp_server_definition(
        server_id="p7-l2-no-surface",
        display_name="P7 L2 no surface",
        default_capability="p7.l2",
        bindings=(McpCatalogToolBinding(binding_id="known", tool_id="p7.l2.known"),),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        compose_mcp_server_from_definition(definition, _deps(tool_registry=tool_registry))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_DEF_INVALID


# --------------------------------------------------------------------------------------
# MCP-ACC-04: exposure policy allowlist on composed host builtins
# --------------------------------------------------------------------------------------


async def _echo_handler(message: str = "") -> dict[str, object]:
    return {"echo": message}


def _three_binding_definition(policy: McpExposurePolicy | None, **overrides: object):  # type: ignore[no-untyped-def]
    bindings = overrides.pop(
        "bindings",
        (
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
            McpHostCustomBinding(binding_id="qual_echo", tool_name="qual_echo"),
        ),
    )
    return build_mcp_server_definition(
        server_id="p7-l2-exposure",
        display_name="P7 L2 exposure",
        default_capability="p7.l2",
        bindings=bindings,  # type: ignore[arg-type]
        exposure_policy=policy,
    )


@pytest.mark.asyncio
async def test_acc_04_allowlist_limits_host_builtins_in_list_tools() -> None:
    definition = _three_binding_definition(
        McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "qual_echo")),
    )
    mcp = compose_mcp_server_from_definition(definition, _deps(custom={"qual_echo": _echo_handler}))
    assert await _tool_names(mcp) == {"list_agents", "qual_echo"}


@pytest.mark.asyncio
async def test_acc_04_non_allowlisted_builtin_cannot_be_called() -> None:
    host = _recording_host()
    definition = _three_binding_definition(
        McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "qual_echo")),
    )
    mcp = compose_mcp_server_from_definition(
        definition,
        _deps(host=host, custom={"qual_echo": _echo_handler}),
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await client.call_tool("run_agent", {"message": "must not run"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_04_default_policy_exposes_every_binding() -> None:
    definition = _three_binding_definition(None)
    mcp = compose_mcp_server_from_definition(definition, _deps(custom={"qual_echo": _echo_handler}))
    assert await _tool_names(mcp) == _REFERENCE_CORE_TOOLS


@pytest.mark.asyncio
async def test_acc_04_disabled_policy_exposes_nothing_and_blocks_calls() -> None:
    host = _recording_host()
    definition = _three_binding_definition(
        McpExposurePolicy(enabled=False, allowed_binding_ids=("list_agents", "run_agent", "qual_echo")),
    )
    mcp = compose_mcp_server_from_definition(
        definition,
        _deps(host=host, custom={"qual_echo": _echo_handler}),
    )
    async with Client(mcp) as client:
        assert await client.list_tools() == []
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await client.call_tool("run_agent", {"message": "must not run"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_04_empty_allowlist_fails_closed_to_no_tools() -> None:
    definition = _three_binding_definition(McpExposurePolicy(enabled=True, allowed_binding_ids=()))
    mcp = compose_mcp_server_from_definition(definition, _deps(custom={"qual_echo": _echo_handler}))
    assert await _tool_names(mcp) == set()


@pytest.mark.asyncio
async def test_acc_04_hidden_binding_is_not_discoverable_even_when_allowlisted() -> None:
    definition = _three_binding_definition(
        None,
        bindings=(
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(
                binding_id="run_agent",
                builtin=McpHostBuiltinId.RUN_AGENT,
                visibility=McpBindingVisibility.HIDDEN,
            ),
        ),
    )
    mcp = compose_mcp_server_from_definition(definition, _deps())
    assert await _tool_names(mcp) == {"list_agents"}


def test_acc_04_allowlist_referencing_unknown_binding_rejected_at_composition() -> None:
    definition = _three_binding_definition(
        McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents", "no_such_binding")),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        compose_mcp_server_from_definition(definition, _deps(custom={"qual_echo": _echo_handler}))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN


@pytest.mark.asyncio
async def test_acc_04_catalog_allowlist_filters_list_and_describe_surfaces() -> None:
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "tool.keep")
    _register_tool(tool_registry, "tool.drop")
    definition = build_mcp_server_definition(
        server_id="p7-l2-catalog-allowlist",
        display_name="P7 L2 catalog allowlist",
        default_capability="p7.l2",
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
    mcp = compose_mcp_server_from_definition(definition, _deps(tool_registry=tool_registry))
    async with Client(mcp) as client:
        listed = await client.call_tool("list_catalog_tools", {})
        kept = await client.call_tool("describe_catalog_tool", {"tool_id": "tool.keep"})
        dropped = await client.call_tool("describe_catalog_tool", {"tool_id": "tool.drop"})
    assert {item["name"] for item in listed.data} == {"tool.keep"}
    assert kept.data.get("name") == "tool.keep"
    assert dropped.data == {"error": "unknown_tool:tool.drop"}


# --------------------------------------------------------------------------------------
# MCP-ACC-16 (L2 part): retired binding composes, is absent, call_tool fails closed
# --------------------------------------------------------------------------------------


def _definition_with_retired_run_agent():  # type: ignore[no-untyped-def]
    return build_mcp_server_definition(
        server_id="p7-l2-retire",
        display_name="P7 L2 retire",
        default_capability="p7.l2",
        bindings=(
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(
                binding_id="run_agent",
                builtin=McpHostBuiltinId.RUN_AGENT,
                retired=True,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_acc_16_retired_binding_composes_and_is_absent_from_list_tools() -> None:
    mcp = compose_mcp_server_from_definition(_definition_with_retired_run_agent(), _deps())
    assert await _tool_names(mcp) == {"list_agents"}


@pytest.mark.asyncio
async def test_acc_16_retired_binding_call_fails_closed_without_executing() -> None:
    host = _recording_host()
    mcp = compose_mcp_server_from_definition(_definition_with_retired_run_agent(), _deps(host=host))
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await client.call_tool("run_agent", {"message": "must not run"})
    host.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_acc_16_retired_custom_binding_has_no_redirect_to_successor() -> None:
    calls: list[str] = []

    async def echo_v2(message: str = "") -> dict[str, object]:
        calls.append(message)
        return {"echo": message, "version": 2}

    definition = build_mcp_server_definition(
        server_id="p7-l2-retire-custom",
        display_name="P7 L2 retire custom",
        default_capability="p7.l2",
        bindings=(
            McpHostCustomBinding(binding_id="echo_v1", tool_name="qual_echo", retired=True),
            McpHostCustomBinding(binding_id="echo_v2", tool_name="qual_echo_v2"),
        ),
    )
    # The retired binding needs no handler: it is never materialized on the server.
    mcp = compose_mcp_server_from_definition(definition, _deps(custom={"qual_echo_v2": echo_v2}))
    assert await _tool_names(mcp) == {"qual_echo_v2"}
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="(?i)unknown tool"):
            await client.call_tool("qual_echo", {"message": "old caller"})
        assert calls == []
        ok = await client.call_tool("qual_echo_v2", {"message": "new caller"})
    assert ok.data == {"echo": "new caller", "version": 2}
    assert calls == ["new caller"]


def test_acc_16_retired_binding_resolve_for_invocation_reports_stable_code() -> None:
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        resolve_binding_for_invocation(_definition_with_retired_run_agent(), "run_agent")
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_RETIRED
    assert exc.value.code.value == "MCP_BINDING_RETIRED"


@pytest.mark.asyncio
async def test_acc_16_unretired_recomposition_restores_binding() -> None:
    retired = _definition_with_retired_run_agent()
    active = retired.model_copy(
        update={
            "bindings": tuple(binding.model_copy(update={"retired": False}) for binding in retired.bindings),
        },
    )
    before = compose_mcp_server_from_definition(retired, _deps())
    after = compose_mcp_server_from_definition(active, _deps())
    assert await _tool_names(before) == {"list_agents"}
    assert await _tool_names(after) == {"list_agents", "run_agent"}


@pytest.mark.asyncio
async def test_acc_16_retired_catalog_binding_removed_from_catalog_surface() -> None:
    tool_registry = ToolRegistry()
    _register_tool(tool_registry, "tool.live")
    _register_tool(tool_registry, "tool.old")
    definition = build_mcp_server_definition(
        server_id="p7-l2-retire-catalog",
        display_name="P7 L2 retire catalog",
        default_capability="p7.l2",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="list_catalog_tools",
                builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
            ),
            McpHostBuiltinBinding(
                binding_id="describe_catalog_tool",
                builtin=McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
            ),
            McpCatalogToolBinding(binding_id="live", tool_id="tool.live"),
            McpCatalogToolBinding(binding_id="old", tool_id="tool.old", retired=True),
        ),
    )
    mcp = compose_mcp_server_from_definition(definition, _deps(tool_registry=tool_registry))
    async with Client(mcp) as client:
        listed = await client.call_tool("list_catalog_tools", {})
        old = await client.call_tool("describe_catalog_tool", {"tool_id": "tool.old"})
    assert {item["name"] for item in listed.data} == {"tool.live"}
    assert old.data == {"error": "unknown_tool:tool.old"}
