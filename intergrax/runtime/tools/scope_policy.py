# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class ToolScopePolicy(Protocol):
    """
    Defines authorization contract for tool execution.

    This contract is evaluated by RuntimeToolInvoker
    before executing a tool.

    Implementations must be pure and side-effect free.
    """

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        """
        Returns True if given agent is allowed to execute given tool.
        """
        ...


class StaticToolScopePolicy:
    """
    Simple static scope policy.

    Allows execution only for tools explicitly listed in allowed_tools.
    """

    def __init__(self, *, allowed_tools: set[str]) -> None:
        self._allowed_tools = frozenset(allowed_tools)

    def is_allowed(self, *, agent_id: str, tool_id: str) -> bool:
        # agent_id currently unused in static policy
        # kept for future extensibility
        return tool_id in self._allowed_tools
