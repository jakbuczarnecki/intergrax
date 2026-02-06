# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pydantic import BaseModel

from intergrax.tools.contracts import ToolContract
from intergrax.tools.tool_executor import ToolHandler

@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """
    Runtime binding between a formal ToolContract and its handler implementation.
    """
    contract: ToolContract
    handler: ToolHandler[BaseModel, BaseModel]


class ToolRegistry:
    """
    Runtime-owned tool registry.

    - keyed strictly by tool_id
    - immutable contracts
    - handlers are execution-only (no enforcement)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, contract: ToolContract, handler: ToolHandler[BaseModel, BaseModel]) -> None:
        if contract.tool_id in self._tools:
            raise ValueError(f"Tool already registered: {contract.tool_id}")
        self._tools[contract.tool_id] = RegisteredTool(contract=contract, handler=handler)

    def get(self, tool_id: str) -> RegisteredTool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"Tool not registered: {tool_id}") from exc

    def has(self, tool_id: str) -> bool:
        return tool_id in self._tools
