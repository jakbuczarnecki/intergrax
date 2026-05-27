# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry


class RegistryToolExecutor:
    """Lookup tool handlers from a :class:`ToolRegistry` (Tier-1 default executor)."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        registered = self._registry.get(request.tool_id)
        return registered.handler.execute(request)
