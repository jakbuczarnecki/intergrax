# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolScopeViolationError(RuntimeError):
    """
    Raised when a tool execution attempt violates ToolScopePolicy.

    This is a hard ENGINE-level capability boundary breach.
    It must be treated as non-retryable.
    """

    run_id: str
    agent_id: str
    tool_id: str

    def __str__(self) -> str:
        return (
            f"Tool '{self.tool_id}' is not allowed for agent "
            f"'{self.agent_id}' (run_id={self.run_id})."
        )
