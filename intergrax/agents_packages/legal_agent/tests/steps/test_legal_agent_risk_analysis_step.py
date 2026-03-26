# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalRiskAnalysisStep + Ollama structured output.

If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest


from intergrax.agents_packages.legal_agent.domain.legal_agent_state import Clause
from intergrax.agents_packages.legal_agent.steps.legal_risk_analysis_step import (
    LegalRiskAnalysisStep,
)
from intergrax.agents_packages.legal_agent.tests.support.step_runtime import build_legal_ollama_runtime_state
from testing_support.builder import require_ollama_reachable

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_risk_analysis_populates_legal_checks() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-risk-1",
    )

    agent_state.clauses = [
        Clause(
            id="c_risk_1",
            text="The supplier shall not be liable for any indirect or consequential damages.",
            category="liability",
            is_sensitive=True,
        ),
    ]

    await LegalRiskAnalysisStep().run_step(state=state, agent_state=agent_state)

    assert len(agent_state.legal_checks) > 0
    assert any(c.clause_id == "c_risk_1" for c in agent_state.legal_checks)
