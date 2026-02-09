# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor

class RegistryToolExecutor(ToolExecutor):
    """
    Minimal ToolExecutor implementation that routes to handlers registered in ToolRegistry.
    No enforcement here (runtime invoker owns enforcement + trace + mapping).
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        reg = self._registry.get(request.tool_id)
        return reg.handler.execute(request)
