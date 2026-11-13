# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the intergrax_assistant_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_application_server import McpIdentityMode, compose_application_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def build_intergrax_assistant_mcp_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    route_prefix: str,
    tool_registry: object | None = None,
    identity_mode: McpIdentityMode = McpIdentityMode.HARNESS_BOUND,
) -> FastMCP:
    """MCP tools mirror the assistant HTTP API (canonical host task execution)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    tool_reg = tool_registry if isinstance(tool_registry, ToolRegistry) else None
    return compose_application_mcp_server(
        display_name="Intergrax Assistant MCP",
        host_execution=host_execution,
        registry=registry,
        default_capability="platform.assist",
        tool_registry=tool_reg,
        identity_mode=identity_mode,
    )
