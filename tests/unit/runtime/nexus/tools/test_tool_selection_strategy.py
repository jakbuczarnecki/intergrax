# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-5 — ToolSelectionStrategy acceptance tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
    strategy_for_mode,
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


def _registry_with(*tool_ids: str) -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id in tool_ids:
        registry.register(
            tools_agent_make_contract(tool_id, _In, _Out),
            _EchoHandler(),
        )
    return registry


def test_static_mode_uses_plan_tool_ids() -> None:
    registry = _registry_with("alpha.tool", "beta.tool")
    ctx = ToolSelectionContext(
        registry=registry,
        query="alpha",
        plan_allowed_tool_ids=("alpha.tool",),
    )
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.STATIC, ctx)
    assert ids == ("alpha.tool",)


def test_keyword_top_k_alias_matches_retrieval_top_k() -> None:
    registry = _registry_with("rag.retrieve", "jira.search_tasks", "notify.send")
    ctx = ToolSelectionContext(registry=registry, query="search jira tasks", top_k=1)
    retrieval = strategy_for_mode(ToolSelectionMode.RETRIEVAL_TOP_K).select_tool_ids(ctx)
    keyword = strategy_for_mode(ToolSelectionMode.KEYWORD_TOP_K).select_tool_ids(ctx)
    assert retrieval == keyword == ("jira.search_tasks",)


def test_retrieval_top_k_ranks_by_query_overlap() -> None:
    registry = _registry_with("rag.retrieve", "jira.search_tasks", "notify.send")
    ctx = ToolSelectionContext(registry=registry, query="search jira tasks", top_k=1)
    ids = strategy_for_mode(ToolSelectionMode.RETRIEVAL_TOP_K).select_tool_ids(ctx)
    assert ids == ("jira.search_tasks",)


def test_full_catalog_mode_returns_no_filter() -> None:
    registry = _registry_with("rag.retrieve")
    ctx = ToolSelectionContext(registry=registry, query="anything")
    ids = strategy_for_mode(ToolSelectionMode.FULL_CATALOG).select_tool_ids(ctx)
    assert ids is None


def test_plan_and_strategy_intersection() -> None:
    registry = _registry_with("alpha.tool", "beta.tool", "gamma.tool")
    ctx = ToolSelectionContext(
        registry=registry,
        query="gamma search",
        plan_allowed_tool_ids=("alpha.tool", "gamma.tool"),
        top_k=1,
    )
    ids = resolve_planner_allowed_tool_ids(ToolSelectionMode.RETRIEVAL_TOP_K, ctx)
    assert ids == ("gamma.tool",)
