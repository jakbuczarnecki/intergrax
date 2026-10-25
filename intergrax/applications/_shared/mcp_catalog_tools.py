# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional catalog tool mount for FastMCP servers (Phase O.6)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.mcp_result_bounds import bound_mcp_structured_result, emit_bounded_mcp_list
from intergrax.tools.registry.runtime import ToolRegistry


def mount_catalog_tools_on_mcp(
    mcp: FastMCP,
    registry: ToolRegistry,
    *,
    include_invoke_stub: bool = True,
    catalog_invocation: McpCatalogInvocationComposition | None = None,
) -> None:
    """
    Register catalog tool schemas on an MCP server.

    When ``include_invoke_stub`` is True, adds ``list_catalog_tools`` and
    ``describe_catalog_tool`` introspection helpers.

    When ``catalog_invocation`` is set, adds ``invoke_catalog_tool`` routed through
    the host-injected :class:`~intergrax.contracts.execution_bound_catalog_tool_invocation.ExecutionBoundCatalogToolInvoker`.
    """
    if include_invoke_stub:

        @mcp.tool
        def list_catalog_tools() -> list[dict[str, Any]]:
            """List enabled catalog tools (tool_id, description, input schema)."""
            payload = to_mcp_tools(registry)
            return emit_bounded_mcp_list(payload)

        @mcp.tool
        def describe_catalog_tool(tool_id: str) -> dict[str, Any]:
            """Return MCP schema for one catalog tool_id."""
            if not registry.has(tool_id):
                payload = {"error": f"unknown_tool:{tool_id}"}
                bounded, truncated = bound_mcp_structured_result(payload)
                if truncated:
                    return {"error": "mcp_payload_byte_limit_exceeded", "tool_id": tool_id}
                return bounded if isinstance(bounded, dict) else {"error": "mcp_payload_byte_limit_exceeded"}
            registered = registry.get(tool_id)
            items = to_mcp_tools([registered.contract])
            payload = items[0] if items else {"error": f"unknown_tool:{tool_id}"}
            bounded, truncated = bound_mcp_structured_result(payload)
            if truncated:
                return {"error": "mcp_payload_byte_limit_exceeded", "tool_id": tool_id}
            return bounded if isinstance(bounded, dict) else {"error": "mcp_payload_byte_limit_exceeded"}

        if catalog_invocation is not None:

            @mcp.tool
            async def invoke_catalog_tool(
                tool_id: str,
                task_id: str,
                input: dict[str, Any] | None = None,
                run_id: str | None = None,
                step_id: str | None = None,
                tenant_id: str | None = None,
            ) -> dict[str, Any]:
                """Invoke one catalog Tool under an active execution binding (canonical ToolRuntime)."""
                return await execute_mcp_catalog_tool_invocation(
                    catalog_invocation,
                    tool_id=tool_id,
                    input_payload=input,
                    task_id=task_id,
                    run_id=run_id,
                    step_id=step_id,
                    asserted_tenant_id=tenant_id,
                )
