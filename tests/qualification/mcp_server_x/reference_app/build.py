# © Artur Czarnecki. All rights reserved.

"""Build the MCP-SERVER-X P6 reference qualification FastMCP server (executable contract)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_developer_api import (
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpHostCustomBinding,
    McpServerCompositionDeps,
    build_mcp_server_definition,
    compose_mcp_server_from_definition,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.tools.registry.runtime import ToolRegistry

REFERENCE_SERVER_ID = "mcp-server-x-p6-reference"
REFERENCE_DISPLAY_NAME = "MCP-SERVER-X P6 Reference"
REFERENCE_CAPABILITY = "mcp.p6.reference"
REFERENCE_TENANT = "mcp-p6-reference"
REFERENCE_USER = "mcp-p6-reference-user"


def build_mcp_server_x_reference_qualification_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    tool_registry: ToolRegistry | None = None,
    custom_echo: Callable[..., Awaitable[dict[str, object]]] | None = None,
) -> FastMCP:
    """Reference server: host builtins + optional catalog surface + qual echo custom binding."""
    catalog_tool_ids: list[str] = []
    if tool_registry is not None:
        catalog_tool_ids = [item.contract.tool_id for item in tool_registry.list()]

    bindings: list = [
        McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
        McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
        McpHostCustomBinding(binding_id="qual_echo", tool_name="qual_echo"),
    ]
    if catalog_tool_ids:
        bindings.extend(
            [
                McpHostBuiltinBinding(
                    binding_id="list_catalog_tools",
                    builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
                ),
                McpHostBuiltinBinding(
                    binding_id="describe_catalog_tool",
                    builtin=McpHostBuiltinId.DESCRIBE_CATALOG_TOOL,
                ),
                McpHostBuiltinBinding(
                    binding_id="invoke_catalog_tool",
                    builtin=McpHostBuiltinId.INVOKE_CATALOG_TOOL,
                ),
            ],
        )

    definition = build_mcp_server_definition(
        server_id=REFERENCE_SERVER_ID,
        display_name=REFERENCE_DISPLAY_NAME,
        default_capability=REFERENCE_CAPABILITY,
        default_tenant_id=REFERENCE_TENANT,
        default_user_id=REFERENCE_USER,
        bindings=tuple(bindings),
    )
    if custom_echo is None:

        async def _default_echo(message: str = "") -> dict[str, object]:
            return {"echo": message, "server_id": REFERENCE_SERVER_ID}

        custom_echo = _default_echo

    deps = McpServerCompositionDeps(
        host_execution=host_execution,
        registry=registry,
        tool_registry=tool_registry,
        custom_host_tools={"qual_echo": custom_echo},
    )
    return compose_mcp_server_from_definition(definition, deps)
