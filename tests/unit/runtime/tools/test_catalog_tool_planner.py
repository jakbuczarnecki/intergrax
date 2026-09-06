# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.nexus.tools.native_planner_action_context import NativePlannerRound
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tools.tool_planner_protocol import IterativeToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


def _empty_native_round() -> NativePlannerRound:
    return NativePlannerRound(
        response=LLMAdapterResponse(content="done", tool_calls=()),
        business_tool_calls=(),
        tool_plan=ToolCallPlan(calls=[]),
        action_context=None,
    )


pytestmark = pytest.mark.gate


def test_catalog_tool_planner_module_does_not_import_tools_agent() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source_path = repo_root / "intergrax" / "runtime" / "nexus" / "tools" / "catalog_tool_planner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "tools_agent" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "tools_agent" not in node.module


def test_catalog_tool_planner_exposes_llm_from_service() -> None:
    assert hasattr(CatalogToolPlanner, "from_registry")
    assert hasattr(CatalogToolPlanner, "plan_tools")


def test_catalog_tool_planner_delegates_plan_native_round() -> None:
    service = MagicMock(spec=ToolPlanningService)
    expected = _empty_native_round()
    service.plan_native_round.return_value = expected
    planner = CatalogToolPlanner(_service=service)
    messages = [ChatMessage(role="user", content="hi")]

    assert isinstance(planner, IterativeToolPlannerProtocol)
    assert planner.plan_native_round(messages, run_id="run-1") == expected
    service.plan_native_round.assert_called_once_with(
        messages,
        allowed_tool_ids=None,
        run_id="run-1",
        tool_choice=None,
        protocol_config=None,
    )


def test_iterative_tool_planner_protocol_exposes_minimal_plan_native_round() -> None:
    sig = inspect.signature(IterativeToolPlannerProtocol.plan_native_round)
    param_names = list(sig.parameters)
    assert param_names == [
        "self",
        "messages",
        "allowed_tool_ids",
        "run_id",
        "tool_choice",
        "protocol_config",
    ]
    assert "prepared_tools_schema" not in param_names
    assert "prepared_tools_schema_hash" not in param_names
    assert "prepared_messages_hash" not in param_names
