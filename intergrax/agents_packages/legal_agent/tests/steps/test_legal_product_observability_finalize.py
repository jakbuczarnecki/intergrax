# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Unit tests: product observability fields on RuntimeAnswer.route.extra (Etap 7)."""

from __future__ import annotations

import pytest

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.domain.legal_tool_plan import LegalToolPlan
from intergrax.agents_packages.legal_agent.domain.legal_product_observability import (
    LegalProductObservability,
)
from intergrax.agents_packages.legal_agent.steps.legal_finalize_answer_step import (
    FinalAnswerModel,
    LegalFinalizeAnswerStep,
)
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def _state_with_fake_llm(
    *,
    run_id: str,
    fake: FakeLLMAdapter,
) -> tuple[RuntimeState, LegalAgentState]:
    session_manager = SessionManager(storage=InMemorySessionStorage())
    # enable_rag=False: RuntimeConfig.validate() requires embedding+vectorstore when True;
    # finalize still records raw nexus flags in legal_product_obs_v1 for hosts.
    runtime_config = RuntimeConfig(
        llm_adapter=fake,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
        tenant_id="t-obs",
        workspace_id="ws-obs",
        tools_mode="off",
    )
    context = RuntimeContext.build(
        config=runtime_config,
        session_manager=session_manager,
        ingestion_service=None,
    )
    request = RuntimeRequest(
        agent_id="legal-agent-obs",
        user_id="u-obs",
        session_id="s-obs",
        message="test",
        tenant_id="t-obs",
        workspace_id="ws-obs",
    )
    state = RuntimeState(
        context=context,
        request=request,
        run_id=run_id,
        llm_usage_tracker=LLMUsageTracker(run_id=run_id),
    )
    state.configure_llm_tracker()
    legal_config = LegalAgentConfig(
        session_manager=session_manager,
        llm_adapter=fake,
        production_mode=False,
        enable_rag=False,
    )
    agent_state = LegalAgentState(config=legal_config)
    return state, agent_state


@pytest.mark.asyncio
async def test_finalize_populates_legal_product_obs_route_extra() -> None:
    fake = FakeLLMAdapter(
        fake_structured_data=FinalAnswerModel(
            answer="Deterministic finalize answer for observability test."
        ),
    )
    state, agent_state = _state_with_fake_llm(run_id="run-obs-1", fake=fake)
    agent_state.legal_dynamic_loop_waves = 2
    agent_state.clause_extraction_retrieval_outcome = "hits"
    agent_state.legal_run_evaluator_degraded = True
    agent_state.last_legal_tool_plan = LegalToolPlan(
        intent="rag",
        confidence=0.88,
        use_rag=True,
        use_tools=False,
        use_websearch=False,
    )
    state.used_rag = True
    state.used_tools = False
    state.used_websearch = False

    await LegalFinalizeAnswerStep().run_step(state=state, agent_state=agent_state)

    assert state.runtime_answer is not None
    extra = state.runtime_answer.route.extra
    assert LegalProductObservability.ROUTE_EXTRA_KEY in extra
    obs = extra[LegalProductObservability.ROUTE_EXTRA_KEY]
    assert obs["schema"] == LegalProductObservability.SCHEMA_ID
    assert obs["loop_waves"] == 2
    assert obs["finalize_empty_fallback"] is False
    assert obs["clause_retrieval_outcome"] == "hits"
    assert obs["evaluator_degraded"] is True
    assert obs["tool_plan_post_governance"] is not None
    assert obs["tool_plan_post_governance"]["intent"] == "rag"
    assert obs["nexus_flags"]["used_rag"] is True
    # RouteInfo.used_rag is gated by RuntimeConfig.enable_rag (Tier-1); obs keeps raw state.used_rag.
    assert state.runtime_answer.route.used_rag is False
    assert state.runtime_answer.route.used_tools is False
    assert state.runtime_answer.stats.input_tokens is not None


@pytest.mark.asyncio
async def test_finalize_product_obs_reports_empty_fallback() -> None:
    fake = FakeLLMAdapter(fake_structured_data=FinalAnswerModel(answer="   "))
    state, agent_state = _state_with_fake_llm(run_id="run-obs-2", fake=fake)
    await LegalFinalizeAnswerStep().run_step(state=state, agent_state=agent_state)
    obs = state.runtime_answer.route.extra[LegalProductObservability.ROUTE_EXTRA_KEY]
    assert obs["finalize_empty_fallback"] is True
