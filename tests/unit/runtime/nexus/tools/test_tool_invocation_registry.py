# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-24 — invocation pattern entry-point loader tests."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationResult,
    resolve_invocation_pattern,
)
from intergrax.runtime.nexus.tools.tool_invocation_registry import load_tool_invocation_pattern
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan

pytestmark = pytest.mark.unit


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


def test_entry_point_pattern_load(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ep:
        name = "custom_pattern"

        def load(self):
            return _CustomPattern

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_registry.entry_points",
        lambda group=None: [_Ep()],
    )
    loaded = load_tool_invocation_pattern("custom_pattern")
    assert loaded is not None
    assert loaded.pattern_id == "custom_pattern"


def test_resolve_invocation_pattern_prefers_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ep:
        name = "custom_pattern"

        def load(self):
            return _CustomPattern

    monkeypatch.setattr(
        "intergrax.runtime.nexus.tools.tool_invocation_registry.entry_points",
        lambda group=None: [_Ep()],
    )
    resolved = resolve_invocation_pattern(
        mode=ToolInvocationMode.SINGLE_PASS,
        max_iterations=1,
        entry_point_pattern_id="custom_pattern",
    )
    assert resolved.pattern_id == "custom_pattern"
