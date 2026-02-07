# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol, runtime_checkable

from intergrax.tools.registry import ToolRegistry

@runtime_checkable
class ToolProvider(Protocol):
    """
    Production contract: modules register their tools here.

    Runtime does not discover tools.
    Runtime does not inspect agents.
    Tools are explicitly registered.
    """

    def register_tools(self, registry: ToolRegistry) -> None: ...
