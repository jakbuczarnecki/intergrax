# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol


class ToolScopePolicy(Protocol):
    """
    Defines authorization contract for tool execution.

    This contract is evaluated by RuntimeToolInvoker
    before executing a tool.

    Implementations must be pure and side-effect free.
    """

    def is_allowed(self, agent_id: str, tool_id: str) -> bool:
        """
        Returns True if given agent is allowed to execute given tool.
        """
        ...
