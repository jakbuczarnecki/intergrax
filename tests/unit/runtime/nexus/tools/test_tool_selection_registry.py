# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-26 / TOOL-ENG-31 — selection strategy plugin and override tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
    resolve_selection_strategy,
)
from intergrax.runtime.nexus.tools.tool_selection_registry import load_tool_selection_strategy
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


class _PinnedStrategy:
    strategy_id = "pinned"

    def select_tool_ids(self, ctx: ToolSelectionContext) -> Sequence[str] | None:
        _ = ctx
        return ("alpha.tool",)


def _registry_with(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(tools_agent_make_contract(tool_id, _In, _Out), _EchoHandler())
    return registry


def test_instance_override_wins_over_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry_with("alpha.tool", "beta.tool")
    ctx = ToolSelectionContext(registry=registry, query="beta")
    ids = resolve_planner_allowed_tool_ids(
        ToolSelectionMode.RETRIEVAL_TOP_K,
        ctx,
        strategy_override=_PinnedStrategy(),
    )
    assert ids == ("alpha.tool",)


def test_entry_point_strategy_load(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ep:
        name = "custom"
        def load(self):
            return _PinnedStrategy

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_selection_registry.entry_points",
        lambda group=None: [_Ep()],
    )
    loaded = load_tool_selection_strategy("custom")
    assert loaded is not None
    registry = _registry_with("alpha.tool", "beta.tool")
    ctx = ToolSelectionContext(registry=registry, query="beta")
    strategy = resolve_selection_strategy(
        ToolSelectionMode.STATIC,
        ctx,
        entry_point_strategy_id="custom",
    )
    assert strategy.select_tool_ids(ctx) == ("alpha.tool",)
