# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalFinalizeAnswerStep + Ollama structured output.

If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest

from testing_support.builder import require_ollama_reachable

from legal_agent.domain.legal_agent_state import (
    Clause,
    LegalCheck,
    LegalDecision,
    LegalRecommendation,
)
from legal_agent.steps.legal_finalize_answer_step import (
    LegalFinalizeAnswerStep,
)
from legal_agent.tests.support.step_runtime import build_legal_ollama_runtime_state

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_finalize_answer_synthesizes_user_facing_text() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-finalize-1",
        message="Summarize legal posture of this contract excerpt.",
    )

    agent_state.clauses = [
        Clause(
            id="c_fin_1",
            text="Payment due within 30 days.",
            category="payment",
            is_sensitive=False,
        ),
    ]
    agent_state.legal_checks = [
        LegalCheck(
            clause_id="c_fin_1",
            valid=True,
            source="LOW",
            details="Standard payment term.",
        ),
    ]
    agent_state.sensitive_flags = []
    agent_state.recommendations = [
        LegalRecommendation(
            clause_id="c_fin_1",
            action="review",
            priority="LOW",
            recommendation="Confirm payment term aligns with org policy.",
        ),
    ]
    agent_state.decision = LegalDecision(
        status="CONDITIONAL",
        confidence=0.7,
        blocking_issues=["Payment term may need approval if over 30 days."],
        summary="Generally acceptable with minor conditions.",
    )
    agent_state.decision_pre_enforcement_status = "APPROVE"
    agent_state.decision_enforcement_modified = True

    await LegalFinalizeAnswerStep().run_step(state=state, agent_state=agent_state)

    assert state.runtime_answer is not None
    assert len(state.runtime_answer.answer.strip()) > 20
    assert "payment" in state.runtime_answer.answer.lower() or "decision" in state.runtime_answer.answer.lower()
