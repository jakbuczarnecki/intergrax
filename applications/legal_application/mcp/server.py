# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the legal_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_legal_mcp_server(
    *,
    nexus_loop: NexusLoop,
    route_prefix: str,
    default_capability: str = "legal.review",
    tool_registry=None,
) -> FastMCP:
    """MCP tools for the Legal product host (same Nexus loop as HTTP)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict = {
        "name": "Intergrax Legal MCP",
        "nexus_loop": nexus_loop,
        "default_capability": default_capability,
        "default_tenant_id": "legal",
        "default_user_id": "legal-user",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    return build_nexus_mcp_server(**kwargs)
