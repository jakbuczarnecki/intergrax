# © Artur Czarnecki. All rights reserved.

"""FastMCP server coupled to the local_workspace_application FastAPI host."""

from __future__ import annotations

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_nexus_server import build_nexus_mcp_server
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def build_local_workspace_mcp_server(
    *,
    nexus_loop: NexusLoop,
    route_prefix: str,
    tool_registry: object | None = None,
) -> FastMCP:
    """MCP tools mirror the lab HTTP API (same NexusLoop / UnifiedTaskRunner)."""
    _ = route_prefix
    from intergrax.tools.registry.runtime import ToolRegistry

    kwargs: dict[str, object] = {
        "name": "Local Workspace MCP",
        "nexus_loop": nexus_loop,
        "default_capability": "local_indexer.basic",
    }
    if isinstance(tool_registry, ToolRegistry):
        kwargs["tool_registry"] = tool_registry
    return build_nexus_mcp_server(**kwargs)
