# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-24 — invocation pattern entry-point loader tests."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence

import pytest

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationResult,
    resolve_invocation_pattern,
)
from intergrax.runtime.nexus.tools.tool_invocation_registry import (
    list_tool_invocation_pattern_ids,
    load_tool_invocation_pattern,
)
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan

pytestmark = pytest.mark.unit


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


class _CustomPattern:
    @property
    def pattern_id(self) -> str:
        return "custom_pattern"

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: object,
    ) -> ToolInvocationResult:
        _ = state, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationResult(pattern_id="custom_pattern", stop_reason="empty_tool_calls")


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "custom_pattern",
                f"{__name__}:_CustomPattern",
                "intergrax.tool_invocation_patterns",
            )
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)


def test_entry_point_pattern_load(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch)
    loaded = load_tool_invocation_pattern("custom_pattern")
    assert loaded is not None
    assert loaded.pattern_id == "custom_pattern"


def test_resolve_invocation_pattern_prefers_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch)
    resolved = resolve_invocation_pattern(
        mode=ToolInvocationMode.SINGLE_PASS,
        max_iterations=1,
        entry_point_pattern_id="custom_pattern",
    )
    assert resolved.pattern_id == "custom_pattern"


def test_load_tool_invocation_pattern_lazy_lookup_skips_unrelated_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "a_pattern",
                f"{__name__}:_CustomPattern",
                "intergrax.tool_invocation_patterns",
            ),
            _EntryPoint(
                "z_pattern",
                f"{__name__}:_CustomPattern",
                "intergrax.tool_invocation_patterns",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    load_calls: list[str] = []

    def _tracking_load(value: str) -> object:
        from intergrax.core.plugins.discovery import load_entry_point_value

        load_calls.append(value)
        return load_entry_point_value(value)

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_registry.load_entry_point_value",
        _tracking_load,
    )

    loaded = load_tool_invocation_pattern("z_pattern")

    assert loaded is not None
    assert loaded.pattern_id == "custom_pattern"
    assert load_calls == [f"{__name__}:_CustomPattern"]
    assert list_tool_invocation_pattern_ids() == ("a_pattern", "z_pattern")
