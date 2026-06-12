# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-10 — adaptive tool mode resolver tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config_types import ToolInvocationMode, ToolSelectionMode
from intergrax.runtime.nexus.tools.adaptive_tool_mode_resolver import recommend_tool_modes
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from testing_support.builder import tools_agent_make_contract

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _EchoHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(result=1)


def _registry_with(count: int) -> ToolRegistry:
    registry = ToolRegistry()
    for index in range(count):
        registry.register(
            tools_agent_make_contract(f"tool.{index}", _In, _Out),
            _EchoHandler(),
        )
    return registry


def test_large_catalog_recommends_semantic() -> None:
    rec = recommend_tool_modes(registry=_registry_with(90), query="find issue")
    assert rec.tool_selection_mode is ToolSelectionMode.SEMANTIC
    assert rec.tool_invocation_mode is ToolInvocationMode.SINGLE_PASS


def test_small_catalog_recommends_full() -> None:
    rec = recommend_tool_modes(registry=_registry_with(5), query="find issue")
    assert rec.tool_selection_mode is ToolSelectionMode.FULL_CATALOG
