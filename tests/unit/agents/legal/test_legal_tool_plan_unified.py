# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from legal.domain.legal_tool_plan import LegalToolPlan
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID

pytestmark = pytest.mark.unit


def test_legal_tool_plan_syncs_tool_ids_from_booleans() -> None:
    plan = LegalToolPlan(
        intent="rag",
        confidence=0.9,
        use_rag=True,
        use_websearch=False,
        use_tools=False,
    )
    assert RAG_RETRIEVE_TOOL_ID in plan.tool_ids
    assert plan.resolved_tool_ids() == plan.tool_ids


def test_legal_tool_plan_syncs_booleans_from_tool_ids() -> None:
    plan = LegalToolPlan(
        intent="websearch",
        confidence=0.8,
        tool_ids=[WEBSEARCH_QUERY_TOOL_ID],
        use_rag=False,
        use_websearch=False,
        use_tools=False,
    )
    assert plan.use_websearch is True
    assert WEBSEARCH_QUERY_TOOL_ID in plan.resolved_tool_ids()
