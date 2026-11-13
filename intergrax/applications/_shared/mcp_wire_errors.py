# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single owner of the MCP wire-error shape: ``"<MCP_CODE>: <detail>"`` carried by ``ToolError``."""

from __future__ import annotations

from fastmcp.exceptions import ToolError

from intergrax.contracts.mcp_server_definition import McpServerDefinitionErrorCode


def mcp_tool_error(code: McpServerDefinitionErrorCode | str, detail: str = "") -> ToolError:
    """Build the canonical MCP wire error (``isError=true`` on the protocol)."""
    code_text = code.value if isinstance(code, McpServerDefinitionErrorCode) else str(code)
    return ToolError(f"{code_text}: {detail}" if detail else code_text)


__all__ = ["mcp_tool_error"]
