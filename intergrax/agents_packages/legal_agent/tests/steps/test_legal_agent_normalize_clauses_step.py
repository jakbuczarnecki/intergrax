# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalNormalizeClausesStep + Ollama structured output.

If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest

from testing_support.builder import require_ollama_reachable

from intergrax.agents_packages.legal_agent.domain.legal_agent_state import Clause
from intergrax.agents_packages.legal_agent.steps.legal_normalize_clauses_step import (
    LegalNormalizeClausesStep,
)
from intergrax.agents_packages.legal_agent.tests.support.step_runtime import build_legal_ollama_runtime_state

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_normalize_clauses_merges_or_dedupes_with_ollama() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-normalize-1",
    )

    agent_state.clauses = [
        Clause(
            id="clause_a",
            text="The client agrees to pay within 14 days of invoice.",
            category="payment",
            is_sensitive=False,
        ),
        Clause(
            id="clause_b",
            text="Payment shall be made within fourteen (14) days from the invoice date.",
            category="payment",
            is_sensitive=False,
        ),
    ]

    await LegalNormalizeClausesStep().run_step(state=state, agent_state=agent_state)

    assert len(agent_state.clauses) >= 1
    assert all(c.text.strip() for c in agent_state.clauses)
