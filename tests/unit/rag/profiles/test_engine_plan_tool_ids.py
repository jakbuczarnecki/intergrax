# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan, PlanIntent
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID

pytestmark = pytest.mark.gate


def test_engine_plan_resolved_tool_ids_from_use_rag() -> None:
    plan = EnginePlan(
        version="1",
        intent=PlanIntent.GENERIC,
        use_rag=True,
    )
    assert RAG_RETRIEVE_TOOL_ID in plan.resolved_tool_ids()
    assert plan.uses_legacy_rag_flag_only() is True
