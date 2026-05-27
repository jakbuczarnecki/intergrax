# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox runtime constants and policy helpers (Phase F.2, §21, §42.12.2)."""

from __future__ import annotations

from typing import FrozenSet

SANDBOX_FLAG = "sandbox"
SANDBOX_SESSION_ID_KEY = "sandbox_session_id"
SANDBOX_CLEANUP_KEY = "sandbox_cleanup"
SANDBOX_TOOL_NAME = "sandbox.exec"

DEFAULT_SANDBOX_OPERATIONS: FrozenSet[str] = frozenset(
    {"echo", "write_file", "read_file", "list_files"}
)

SANDBOX_REQUIRED_TOOLS: FrozenSet[str] = frozenset(
    {SANDBOX_TOOL_NAME, "code.exec", "browser.run", "script.run"}
)


def requires_sandbox_tool(tool_name: str) -> bool:
    """True when a tool MUST route through sandbox policy (§42.12.2 rule 5)."""
    return tool_name in SANDBOX_REQUIRED_TOOLS
