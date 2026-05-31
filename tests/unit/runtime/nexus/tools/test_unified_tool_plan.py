# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

pytestmark = pytest.mark.unit


def test_tool_invocation_plan_from_legacy_booleans() -> None:
    plan = ToolInvocationPlan.from_legacy(use_rag=True, use_websearch=True)
    assert RAG_RETRIEVE_TOOL_ID in plan.tool_ids
    assert WEBSEARCH_QUERY_TOOL_ID in plan.tool_ids
    assert plan.use_rag is True
    assert plan.use_websearch is True


def test_tool_invocation_plan_from_tool_ids() -> None:
    plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    assert plan.use_rag is True
    assert plan.use_websearch is False


def test_tool_invocation_plan_uses_legacy_booleans_only() -> None:
    legacy = ToolInvocationPlan(use_rag=True)
    assert legacy.uses_legacy_booleans_only() is True
    canonical = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    assert canonical.uses_legacy_booleans_only() is False
