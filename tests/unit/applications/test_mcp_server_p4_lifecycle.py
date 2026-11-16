# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from intergrax.applications._shared.mcp_server_engine import (
    McpServerCompositionDeps,
    compose_mcp_server_from_definition,
    exposed_builtin_tool_names,
)
from intergrax.contracts.mcp_server_definition import (
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpServerDefinitionErrorCode,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    resolve_binding_for_invocation,
    resolve_effective_bindings,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from tests.fixtures.harness_host_task_execution import build_harness_host_task_execution

pytestmark = [pytest.mark.unit]


def test_exposure_disabled_yields_empty_discovery_bindings() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
        ),
        exposure_policy=McpExposurePolicy(enabled=False, allowed_binding_ids=("list_agents", "run_agent")),
    )
    assert resolve_effective_bindings(definition) == ()
    assert exposed_builtin_tool_names(definition) == []


@pytest.mark.asyncio
async def test_exposure_disabled_composes_without_tools() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),),
        exposure_policy=McpExposurePolicy(enabled=False, allowed_binding_ids=("list_agents",)),
    )
    registry = AgentRegistry()
    deps = McpServerCompositionDeps(
        host_execution=build_harness_host_task_execution(NexusLoop(registry), orchestration_triggers=frozenset()),
        registry=registry,
    )
    mcp = compose_mcp_server_from_definition(definition, deps)
    from fastmcp import Client

    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert tools == []


def test_retired_binding_absent_from_discovery() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="run_agent",
                builtin=McpHostBuiltinId.RUN_AGENT,
                retired=True,
            ),
        ),
    )
    assert resolve_effective_bindings(definition) == ()


def test_retired_binding_invoke_fails_closed() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="run_agent",
                builtin=McpHostBuiltinId.RUN_AGENT,
                retired=True,
            ),
        ),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        resolve_binding_for_invocation(definition, "run_agent")
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_RETIRED


def test_lab_application_omits_mcp_mount_when_disabled() -> None:
    from intergrax.fastapi_core.config import ApiEnvironment

    from lab_application.host.factory import create_lab_application
    from lab_application.host.settings import LabApplicationSettings

    settings = LabApplicationSettings(
        environment=ApiEnvironment.DEV,
        include_mcp=False,
        include_scheduler=False,
    )
    app = create_lab_application(settings=settings)
    mcp_paths = {getattr(r, "path", None) for r in app.routes if hasattr(r, "path")}
    assert "/mcp" not in mcp_paths
