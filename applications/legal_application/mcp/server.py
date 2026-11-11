# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the legal_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_application_server import compose_application_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def build_legal_mcp_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    route_prefix: str,
    default_capability: str = "legal.review",
    tool_registry=None,
) -> FastMCP:
    """MCP tools for the Legal product host (canonical host task execution)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    tool_reg = tool_registry if isinstance(tool_registry, ToolRegistry) else None
    return compose_application_mcp_server(
        display_name="Intergrax Legal MCP",
        host_execution=host_execution,
        registry=registry,
        default_capability=default_capability,
        default_tenant_id="legal",
        default_user_id="legal-user",
        tool_registry=tool_reg,
    )
