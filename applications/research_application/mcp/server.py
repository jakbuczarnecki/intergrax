# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the research_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import (
    build_nexus_mcp_server,
    execute_mcp_agent_task,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


def build_research_mcp_server(
    *,
    host_execution: HostTaskExecutionPort,
    registry: AgentRegistryRead,
    route_prefix: str,
    tool_registry=None,
) -> FastMCP:
    """MCP tools for research host — includes research pipeline tool."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict[str, object] = {
        "name": "Intergrax Research MCP",
        "host_execution": host_execution,
        "registry": registry,
        "default_capability": "research.pipeline",
        "default_tenant_id": "research",
        "default_user_id": "research-user",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    mcp = build_nexus_mcp_server(**kwargs)

    @mcp.tool
    async def run_research_pipeline(
        message: str,
        tenant_id: str = "research",
        user_id: str = "research-user",
    ) -> dict[str, object]:
        """Run research → summarize pipeline (same as POST .../run)."""
        return await execute_mcp_agent_task(
            host_execution,
            message=message,
            capability="research.pipeline",
            tenant_id=tenant_id,
            user_id=user_id,
            intent="research_summarize",
        )

    return mcp
