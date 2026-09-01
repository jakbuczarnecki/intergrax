# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the dispute_sim_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.execution.host_task import HostTaskExecutionPort


def build_dispute_sim_mcp_server(
    *,
    host_execution: HostTaskExecutionPort,
    route_prefix: str,
    tool_registry: object | None = None,
) -> FastMCP:
    """MCP tools mirror the dispute sim HTTP API (canonical host task execution)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict[str, object] = {
        "name": "Dispute Sim MCP",
        "host_execution": host_execution,
        "default_capability": "dispute_intake.basic",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    return build_nexus_mcp_server(**kwargs)
