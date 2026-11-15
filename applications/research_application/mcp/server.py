# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the research_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP
from intergrax.applications._shared.mcp_application_server import (
    McpIdentityMode,
    compose_application_mcp_server,
    execute_mcp_agent_task_with_resolved_identity,
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
    identity_mode: McpIdentityMode = McpIdentityMode.HARNESS_BOUND,
) -> FastMCP:
    """MCP tools for research host — includes research pipeline tool as host custom binding."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    tool_reg = tool_registry if isinstance(tool_registry, ToolRegistry) else None

    async def run_research_pipeline(
        message: str,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, object]:
        """Run research → summarize pipeline (same as POST .../run)."""
        return await execute_mcp_agent_task_with_resolved_identity(
            host_execution,
            message=message,
            capability="research.pipeline",
            asserted_tenant_id=tenant_id,
            asserted_user_id=user_id,
            default_tenant_id="research",
            default_user_id="research-user",
            identity_mode=identity_mode,
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
        identity_mode=identity_mode,
        extra_bindings=(research_pipeline_binding(),),
        custom_host_tools={"run_research_pipeline": run_research_pipeline},
    )
