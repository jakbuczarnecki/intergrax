# © Artur Czarnecki. All rights reserved.

"""Shared fakes for Nexus tool engine unit tests."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry.runtime import RegisteredTool, ToolRegistry


class StubToolHandler:
    """Minimal handler for invoker tests that only exercise the executor path."""

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        _ = request
        raise RuntimeError("StubToolHandler must not be called when a custom executor is injected.")


def make_registered_tool(contract: ToolContract) -> RegisteredTool:
    return RegisteredTool(contract=contract, handler=StubToolHandler())


class FakeRegistry(ToolRegistry):
    """Single-tool registry for invoker policy/duration tests."""

    def __init__(self, contract: ToolContract) -> None:
        super().__init__()
        self.register(contract, StubToolHandler())
