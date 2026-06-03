# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MCP tool definition export from ``ToolContract`` (Phase O.6)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.runtime import RegisteredTool, ToolRegistry


def contract_to_mcp_tool(contract: ToolContract, *, compact_description: bool = False) -> dict[str, Any]:
    return {
        "name": contract.tool_id,
        "description": contract.llm_description(compact=compact_description),
        "inputSchema": contract.input_schema.model_json_schema(),
        "annotations": {
            "side_effects": contract.side_effects,
            "risk_level": contract.risk_level.value,
            "category": contract.category,
            "injects_context": contract.injects_context,
        },
    }


def to_mcp_tools(
    source: ToolRegistry | Iterable[RegisteredTool] | Iterable[ToolContract],
    *,
    compact_description: bool = False,
) -> list[dict[str, Any]]:
    if isinstance(source, ToolRegistry):
        contracts = [item.contract for item in source.list()]
    else:
        items = list(source)
        if not items:
            return []
        if isinstance(items[0], ToolContract):
            contracts = items  # type: ignore[assignment]
        else:
            contracts = [item.contract for item in items]  # type: ignore[union-attr]
    return [contract_to_mcp_tool(c, compact_description=compact_description) for c in contracts]


export_mcp_tools = to_mcp_tools
