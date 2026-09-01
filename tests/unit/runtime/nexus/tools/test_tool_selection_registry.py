# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-26 / TOOL-ENG-31 — selection strategy plugin and override tests."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from intergrax.core.plugins.discovery import (
    EP_TOOL_SELECTION_STRATEGIES,
    reset_entry_point_spec_cache_for_tests,
)
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
    resolve_selection_strategy,
)
from intergrax.runtime.nexus.tools.tool_selection_registry import (
    list_tool_selection_strategy_ids,
    load_tool_selection_strategy,
)
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


_PINNED_STRATEGY_INSTANCE = _PinnedStrategy()


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


@pytest.fixture(autouse=True)
def _reset_entry_point_spec_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()


def _install_eps(monkeypatch: pytest.MonkeyPatch, entries: list[_EntryPoint]) -> None:
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints(entries))


def _strategy_ep(name: str, attr: str) -> _EntryPoint:
    return _EntryPoint(name, f"{__name__}:{attr}", EP_TOOL_SELECTION_STRATEGIES)


def _registry_with(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(tools_agent_make_contract(tool_id, _In, _Out), _EchoHandler())
    return registry


def test_instance_override_wins_over_mode() -> None:
    registry = _registry_with("alpha.tool", "beta.tool")
    ctx = ToolSelectionContext(registry=registry, query="beta")
    ids = resolve_planner_allowed_tool_ids(
        ToolSelectionMode.RETRIEVAL_TOP_K,
        ctx,
        strategy_override=_PinnedStrategy(),
    )
    assert ids == ("alpha.tool",)


def test_entry_point_strategy_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("custom", "_PinnedStrategy")])
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


def test_load_missing_strategy_returns_none() -> None:
    assert load_tool_selection_strategy("missing-strategy") is None


def test_load_class_entry_point_instantiates(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("custom", "_PinnedStrategy")])
    loaded = load_tool_selection_strategy("custom")
    assert isinstance(loaded, _PinnedStrategy)


def test_load_instance_entry_point_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(monkeypatch, [_strategy_ep("custom", "_PINNED_STRATEGY_INSTANCE")])
    loaded = load_tool_selection_strategy("custom")
    assert loaded is _PINNED_STRATEGY_INSTANCE


def test_list_tool_selection_strategy_ids_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_eps(
        monkeypatch,
        [
            _strategy_ep("beta", "_PinnedStrategy"),
            _strategy_ep("alpha", "_PinnedStrategy"),
        ],
    )
    assert list_tool_selection_strategy_ids() == ("alpha", "beta")
