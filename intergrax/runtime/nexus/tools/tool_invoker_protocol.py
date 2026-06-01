# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Tool invoker protocol for Nexus (Phase Q+-T.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry


@runtime_checkable
class ToolInvokerProtocol(Protocol):
    """Catalog tool invoker surface used by pipeline steps and ToolRuntime."""

    @property
    def registry(self) -> ToolRegistry:
        ...

    def invoke(
        self,
        *,
        state: object,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:
        ...
