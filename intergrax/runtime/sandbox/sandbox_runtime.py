# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sandbox runtime constants and policy helpers (Phase F.2, §21, §42.12.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import FrozenSet

SANDBOX_TOOL_NAME = "sandbox.exec"


class SandboxMetadataKey(StrEnum):
    """Flat metadata keys for sandbox isolation (Phase F.2)."""

    SANDBOX = "sandbox"
    SANDBOX_SESSION_ID = "sandbox_session_id"
    SANDBOX_CLEANUP = "sandbox_cleanup"


SANDBOX_FLAG = SandboxMetadataKey.SANDBOX
SANDBOX_SESSION_ID_KEY = SandboxMetadataKey.SANDBOX_SESSION_ID
SANDBOX_CLEANUP_KEY = SandboxMetadataKey.SANDBOX_CLEANUP

DEFAULT_SANDBOX_OPERATIONS: FrozenSet[str] = frozenset(
    {"echo", "write_file", "read_file", "list_files"}
)

AGENT_BUILDER_SANDBOX_OPERATIONS: FrozenSet[str] = frozenset(
    {
        *DEFAULT_SANDBOX_OPERATIONS,
        "run_python",
        "run_script",
        "browser_fetch",
    }
)

SANDBOX_REQUIRED_TOOLS: FrozenSet[str] = frozenset(
    {SANDBOX_TOOL_NAME, "code.exec", "browser.run", "script.run"}
)


def requires_sandbox_tool(tool_name: str) -> bool:
    """True when a tool MUST route through sandbox policy (§42.12.2 rule 5)."""
    return tool_name in SANDBOX_REQUIRED_TOOLS
