# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional catalog tool mount for FastMCP servers (Phase O.6)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.registry.runtime import ToolRegistry


def mount_catalog_tools_on_mcp(
    mcp: FastMCP,
    registry: ToolRegistry,
    *,
    include_invoke_stub: bool = True,
) -> None:
    """
    Register catalog tool schemas on an MCP server.

    When ``include_invoke_stub`` is True, adds ``list_catalog_tools`` and
    ``describe_catalog_tool`` introspection helpers (no live runtime invoke).
    """
    if include_invoke_stub:

        @mcp.tool
        def list_catalog_tools() -> list[dict[str, Any]]:
            """List enabled catalog tools (tool_id, description, input schema)."""
            return to_mcp_tools(registry)

        @mcp.tool
        def describe_catalog_tool(tool_id: str) -> dict[str, Any]:
            """Return MCP schema for one catalog tool_id."""
            registered = registry.get(tool_id)
            items = to_mcp_tools([registered.contract])
            return items[0] if items else {"error": f"unknown_tool:{tool_id}"}
