# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalRecommendationStep + Ollama structured output.

If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest

from tests._support.builder import require_ollama_reachable

from intergrax.agents_packages.legal_agent.legal_agent_state import (
    Clause,
    LegalCheck,
    SensitiveFlag,
)
from intergrax.agents_packages.legal_agent.steps.legal_recommendation_step import (
    LegalRecommendationStep,
)
from ._legal_agent_step_runtime import build_legal_ollama_runtime_state

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_recommendation_step_produces_recommendations() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-rec-1",
    )

    agent_state.clauses = [
        Clause(
            id="c_rec_1",
            text="Supplier unlimited liability for all claims worldwide.",
            category="liability",
            is_sensitive=True,
        ),
    ]
    agent_state.legal_checks = [
        LegalCheck(
            clause_id="c_rec_1",
            valid=False,
            source="HIGH",
            details="Unlimited liability exposure.",
        ),
    ]
    agent_state.sensitive_flags = [
        SensitiveFlag(
            clause_id="c_rec_1",
            reason="Unlimited liability clause.",
        ),
    ]

    await LegalRecommendationStep().run_step(state=state, agent_state=agent_state)

    assert len(agent_state.recommendations) > 0
    assert any(r.clause_id == "c_rec_1" for r in agent_state.recommendations)
