# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the local_workspace_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def build_local_workspace_mcp_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    route_prefix: str,
    tool_registry: object | None = None,
) -> FastMCP:
    """MCP tools mirror the local workspace HTTP API (canonical host task execution)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    tool_reg = tool_registry if isinstance(tool_registry, ToolRegistry) else None
    return compose_application_mcp_server(
        display_name="Local Workspace MCP",
        host_execution=host_execution,
        registry=registry,
        default_capability="local_indexer.basic",
        tool_registry=tool_reg,
    )
