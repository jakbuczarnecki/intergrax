# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.legal_response_governance_impl import (
    CallableLegalResponseGovernance,
    PassthroughLegalResponseGovernance,
)
from intergrax.agents_packages.legal_agent.legal_shaped_client_response import (
    LegalShapedClientResponse,
    compose_legal_client_answer_text,
)
from intergrax.agents_packages.legal_agent.steps.legal_finalize_answer_step import (
    FinalAnswerModel,
    LegalFinalizeAnswerStep,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def test_compose_legal_client_answer_text_joins_blocks() -> None:
    shaped = LegalShapedClientResponse(
        body="Main.",
        uncertainty_summary="Limits apply.",
        disclaimer_block="Not legal advice.",
    )
    out = compose_legal_client_answer_text(shaped)
    assert "Main." in out
    assert "Limits apply." in out
    assert "Not legal advice." in out
    assert out.index("Main.") < out.index("Limits apply.")


@pytest.mark.asyncio
async def test_finalize_step_applies_response_governance() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter(fake_structured_data=FinalAnswerModel(answer="Draft only."))

    def _gov(
        draft: str,
        st: RuntimeState,
        ag: LegalAgentState,
        lc: LegalAgentConfig,
    ) -> LegalShapedClientResponse:
        assert draft == "Draft only."
        return LegalShapedClientResponse(
            body=draft,
            disclaimer_block="[Product disclaimer]",
            format_version="test.v1",
        )

    cfg = LegalAgentConfig(
        session_manager=sm,
        llm_adapter=llm,
        production_mode=False,
        legal_response_governance=CallableLegalResponseGovernance(_gov),
    )

    rc = RuntimeConfig(
        llm_adapter=llm,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
    )
    ctx = RuntimeContext.build(
        config=rc,
        session_manager=sm,
        ingestion_service=None,
    )
    req = RuntimeRequest(
        agent_id="a",
        user_id="u",
        session_id="s",
        message="m",
        tenant_id="t",
    )
    state = RuntimeState(context=ctx, request=req, run_id="rgov-1")
    state.agent_state = LegalAgentState(config=cfg)

    await LegalFinalizeAnswerStep().run(state)

    assert state.raw_answer == "Draft only."
    assert state.runtime_answer is not None
    assert "[Product disclaimer]" in state.runtime_answer.answer
    assert "Draft only." in state.runtime_answer.answer
    assert state.runtime_answer.route.extra.get("legal_response_governance_applied") is True
    assert state.runtime_answer.route.extra.get("legal_client_response_format_version") == "test.v1"


@pytest.mark.asyncio
async def test_finalize_step_without_governance_uses_draft() -> None:
    sm = build_in_memory_session_manager()
    llm = FakeLLMAdapter(fake_structured_data=FinalAnswerModel(answer="Plain."))
    cfg = LegalAgentConfig(
        session_manager=sm,
        llm_adapter=llm,
        production_mode=False,
    )
    rc = RuntimeConfig(
        llm_adapter=llm,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
    )
    ctx = RuntimeContext.build(config=rc, session_manager=sm, ingestion_service=None)
    state = RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="a",
            user_id="u",
            session_id="s",
            message="m",
            tenant_id="t",
        ),
        run_id="rgov-2",
    )
    state.agent_state = LegalAgentState(config=cfg)
    await LegalFinalizeAnswerStep().run(state)
    assert state.runtime_answer is not None
    assert state.runtime_answer.answer == "Plain."
    assert state.runtime_answer.route.extra.get("legal_response_governance_applied") is False


def test_passthrough_governance_returns_body_only() -> None:
    sm = build_in_memory_session_manager()
    cfg = LegalAgentConfig(session_manager=sm, llm_adapter=FakeLLMAdapter(), production_mode=False)
    gov = PassthroughLegalResponseGovernance()
    # Minimal state/agent_state for port call (unused by passthrough)
    rc = RuntimeConfig(llm_adapter=cfg.llm_adapter, enable_rag=False, production_mode=False)
    ctx = RuntimeContext.build(config=rc, session_manager=sm, ingestion_service=None)
    st = RuntimeState(
        context=ctx,
        request=RuntimeRequest(agent_id="a", user_id="u", session_id="s", message="m", tenant_id="t"),
        run_id="x",
    )
    ag = LegalAgentState(config=cfg)
    out = gov.shape_legal_client_response("Hi", state=st, agent_state=ag, legal_config=cfg)
    assert out.body == "Hi"
    assert out.disclaimer_block == ""
