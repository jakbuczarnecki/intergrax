# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-14 — hierarchical tool selection tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.hierarchical_tool_selector import (
    rank_categories,
    select_tools_hierarchical,
)
from intergrax.runtime.nexus.tools.tool_selection import (
    HierarchicalToolSelectionStrategy,
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 0


class _EchoHandler:
    def execute(self, request: ToolExecutionRequest[_In]) -> _Out:
        _ = request
        return _Out(result=1)


def _contract(tool_id: str, *, category: str, description: str) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=description,
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        category=category,
    )


def _registry_with(*contracts: ToolContract) -> ToolRegistry:
    registry = ToolRegistry()
    for contract in contracts:
        registry.register(contract, _EchoHandler())
    return registry


def test_category_pass_ranks_issue_tracker_before_notify() -> None:
    registry = _registry_with(
        _contract("jira.search", category="issue_tracker", description="search jira tasks"),
        _contract("notify.send", category="notifications", description="send email alert"),
    )
    ranks = rank_categories(registry, "search jira tasks")
    assert ranks[0].category == "issue_tracker"
    assert "jira.search" in ranks[0].tool_ids


def test_tool_pass_returns_issue_tracker_tool() -> None:
    registry = _registry_with(
        _contract("jira.search", category="issue_tracker", description="search jira tasks"),
        _contract("notify.send", category="notifications", description="send email alert"),
    )
    ids = select_tools_hierarchical(registry, "search jira tasks", top_k=1, max_category_passes=1)
    assert ids == ("jira.search",)


def test_hierarchical_strategy_integration() -> None:
    registry = _registry_with(
        _contract("jira.search", category="issue_tracker", description="search jira tasks"),
        _contract("notify.send", category="notifications", description="send email alert"),
    )
    ctx = ToolSelectionContext(
        registry=registry,
        query="search jira tasks",
        top_k=1,
        max_hierarchy_passes=1,
    )
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.HIERARCHICAL, ctx)
    assert ids == ("jira.search",)
    strategy = HierarchicalToolSelectionStrategy(max_category_passes=1)
    strategy.select_tool_ids(ctx)
    assert strategy.last_categories == ("issue_tracker",)
