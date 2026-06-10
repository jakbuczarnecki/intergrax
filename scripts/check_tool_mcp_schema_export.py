#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-11.2 — MCP / function-schema export for shipped tool catalog."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT,):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from intergrax.tools.exporters.mcp import to_mcp_tools
from intergrax.tools.exporters.openai import to_openai_tools
from intergrax.tools.providers.rag.bundle import rag_retrieve_contract
from intergrax.tools.providers.rag.handler import RagRetrieveHandler
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


def main() -> int:
    registry = ToolRegistry()
    contract = rag_retrieve_contract()
    registry.register(contract, RagRetrieveHandler(ToolWiringContext()))

    mcp_tools = to_mcp_tools(registry)
    openai_tools = to_openai_tools(registry)
    if not mcp_tools or not openai_tools:
        print("tool schema export produced empty payload", file=sys.stderr)
        return 1
    if "name" not in mcp_tools[0] or "function" not in openai_tools[0]:
        print("exported schemas missing required keys", file=sys.stderr)
        return 1

    print(f"OK: tool MCP/schema export ({len(mcp_tools)} tools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
