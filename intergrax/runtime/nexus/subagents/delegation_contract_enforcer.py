# © Artur Czarnecki. All rights reserved.

"""Runtime enforcement of per-contract tool allowlists (IDEAL-10.3)."""

from __future__ import annotations

from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy


class DelegationToolPolicyError(ValueError):
    """Raised when a delegated subtask requests a tool outside its allowlist."""


def enforce_subtask_tool_allowlist(
    contract: SubtaskContract,
    requested_tool: str,
    *,
    parent_allowed: tuple[str, ...] | None = None,
) -> None:
    allowlist = tuple(contract.allowed_tools or ())
    if allowlist and not ToolAccessPolicy.is_tool_allowed(requested_tool, allowlist):
        raise DelegationToolPolicyError(
            f"tool {requested_tool!r} not in subtask allowlist {allowlist!r}"
        )
    if parent_allowed is not None and not ToolAccessPolicy.is_tool_allowed(
        requested_tool, parent_allowed
    ):
        raise DelegationToolPolicyError(
            f"tool {requested_tool!r} not allowed by parent policy"
        )
