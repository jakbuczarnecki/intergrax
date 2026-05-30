# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the lab_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_lab_mcp_server(
    *,
    nexus_loop: NexusLoop,
    route_prefix: str,
    default_capability: str = "echo.basic",
    tool_registry=None,
) -> FastMCP:
    """MCP tools mirror ``GET {route_prefix}/agents`` and ``POST {route_prefix}/run``."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict = {
        "name": "Intergrax Lab MCP",
        "nexus_loop": nexus_loop,
        "default_capability": default_capability,
        "default_tenant_id": "lab",
        "default_user_id": "lab-user",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    return build_nexus_mcp_server(**kwargs)
