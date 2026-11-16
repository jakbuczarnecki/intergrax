# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional catalog tool mount for FastMCP servers (Phase O.6)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from intergrax.applications._shared.mcp_catalog_invocation_bridge import (
    McpCatalogInvocationComposition,
    execute_mcp_catalog_tool_invocation,
)
from intergrax.applications._shared.mcp_engine_concurrency import (
    McpEngineConcurrentCallAdmission,
    optional_mcp_call_admission,
)
from intergrax.applications._shared.mcp_harness_principal import require_mcp_caller_principal
from intergrax.applications._shared.mcp_identity_mode import McpIdentityMode
from intergrax.applications._shared.mcp_wire_errors import bound_mcp_result_or_raise, sanitized_tool_errors
from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.registry.runtime import ToolRegistry

_MAX_ECHOED_TOOL_ID_CHARS = 128


def _unknown_tool_payload(tool_id: str) -> dict[str, Any]:
    """Unknown-tool result that never echoes unbounded caller input."""
    return {
        "error": f"unknown_tool:{tool_id[:_MAX_ECHOED_TOOL_ID_CHARS]}",
        "mcp_code": "MCP_CATALOG_TOOL_REFUSED",
    }


def mount_catalog_tools_on_mcp(
    mcp: FastMCP,
    registry: ToolRegistry,
    *,
    include_invoke_stub: bool = True,
    catalog_invocation: McpCatalogInvocationComposition | None = None,
    identity_mode: McpIdentityMode = McpIdentityMode.HARNESS_BOUND,
    call_admission: McpEngineConcurrentCallAdmission | None = None,
) -> None:
    """
    Register catalog tool schemas on an MCP server.

    When ``include_invoke_stub`` is True, adds ``list_catalog_tools`` and
    ``describe_catalog_tool`` introspection helpers.

    When ``catalog_invocation`` is set, adds ``invoke_catalog_tool`` routed through
    the host-injected
    :class:`~intergrax.contracts.execution_bound_catalog_tool_invocation.ExecutionBoundCatalogToolInvoker`.
    """
    if include_invoke_stub:

        @mcp.tool
        async def list_catalog_tools() -> list[dict[str, Any]]:
            """List enabled catalog tools (tool_id, description, input schema)."""
            async with sanitized_tool_errors("list_catalog_tools"):
                require_mcp_caller_principal(identity_mode=identity_mode)
                # registry reads may block (host-provided registries): keep them off the server event loop
                payload = await asyncio.to_thread(to_mcp_tools, registry)
                return bound_mcp_result_or_raise(payload)

        @mcp.tool
        async def describe_catalog_tool(tool_id: str) -> dict[str, Any]:
            """Return MCP schema for one catalog tool_id."""
            async with sanitized_tool_errors("describe_catalog_tool"):
                require_mcp_caller_principal(identity_mode=identity_mode)

                def _describe() -> dict[str, Any]:
                    if not registry.has(tool_id):
                        return _unknown_tool_payload(tool_id)
                    registered = registry.get(tool_id)
                    items = to_mcp_tools([registered.contract])
                    return items[0] if items else _unknown_tool_payload(tool_id)

                return bound_mcp_result_or_raise(await asyncio.to_thread(_describe))

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
                async with optional_mcp_call_admission(call_admission), sanitized_tool_errors("invoke_catalog_tool"):
                    return await execute_mcp_catalog_tool_invocation(
                        catalog_invocation,
                        tool_id=tool_id,
                        input_payload=input,
                        task_id=task_id,
                        run_id=run_id,
                        step_id=step_id,
                        asserted_tenant_id=tenant_id,
                    )
