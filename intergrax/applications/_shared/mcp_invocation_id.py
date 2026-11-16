# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MCP invocation correlation id (MCP-RC MRC-4, OD-MCP-RC-01 D-5).

A per-call, opaque correlation key that ties one MCP tool call to the platform ``task_id`` / ``run_id`` /
``execution_id`` in logs, task metadata and audit evidence. It is **not** an execution authority, is not
derived from tenant or user, and carries no personal data.
"""

from __future__ import annotations

import uuid

MCP_INVOCATION_ID_METADATA_KEY = "mcp_invocation_id"
_PREFIX = "mcpi-"


def mint_mcp_invocation_id() -> str:
    """Return a fresh ``mcpi-<32 hex>`` correlation id."""
    return f"{_PREFIX}{uuid.uuid4().hex}"


__all__ = ["MCP_INVOCATION_ID_METADATA_KEY", "mint_mcp_invocation_id"]
