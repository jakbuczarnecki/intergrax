# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the poc_template_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_poc_template_mcp_server(
    *,
    nexus_loop: NexusLoop,
    route_prefix: str,
) -> FastMCP:
    """MCP tools mirror the lab HTTP API (same NexusLoop / UnifiedTaskRunner)."""
    _ = route_prefix
    return build_nexus_mcp_server(
        name="Poc Template MCP",
        nexus_loop=nexus_loop,
        default_capability="echo.basic",
    )
