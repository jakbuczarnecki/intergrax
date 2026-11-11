# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the research_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_application_server import (
    compose_application_mcp_server,
    execute_mcp_agent_task,
    research_pipeline_binding,
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
    """MCP tools for research host — includes research pipeline tool as host custom binding."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    tool_reg = tool_registry if isinstance(tool_registry, ToolRegistry) else None

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

    return compose_application_mcp_server(
        display_name="Intergrax Research MCP",
        host_execution=host_execution,
        registry=registry,
        default_capability="research.pipeline",
        default_tenant_id="research",
        default_user_id="research-user",
        tool_registry=tool_reg,
        extra_bindings=(research_pipeline_binding(),),
        custom_host_tools={"run_research_pipeline": run_research_pipeline},
    )
