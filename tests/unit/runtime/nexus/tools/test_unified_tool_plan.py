# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.tools.tool_runtime import (
    ToolInvocationPlan,
    plan_includes_rag,
    plan_includes_websearch,
)
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

pytestmark = pytest.mark.unit


def test_tool_invocation_plan_from_tool_ids() -> None:
    plan = ToolInvocationPlan.from_tool_ids([RAG_RETRIEVE_TOOL_ID])
    assert plan.tool_ids == (RAG_RETRIEVE_TOOL_ID,)
    assert plan_includes_rag(plan.tool_ids) is True
    assert plan_includes_websearch(plan.tool_ids) is False


def test_tool_invocation_plan_dedupes_tool_ids() -> None:
    plan = ToolInvocationPlan.from_tool_ids(
        [RAG_RETRIEVE_TOOL_ID, RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID],
    )
    assert plan.tool_ids == (RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID)


def test_tool_invocation_plan_use_tools_flag() -> None:
    plan = ToolInvocationPlan.from_tool_ids([], use_tools=True)
    assert plan.tool_ids == ()
    assert plan.use_tools is True
