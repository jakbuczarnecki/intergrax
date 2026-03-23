# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalPolicyComplianceStep + Ollama structured output.

Uses default organization policy (forbids unlimited liability). If Ollama is
unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest

from tests._support.builder import require_ollama_reachable

from intergrax.agents_packages.legal_agent.legal_agent_state import Clause
from intergrax.agents_packages.legal_agent.steps.legal_policy_compliance_step import (
    LegalPolicyComplianceStep,
)
from .._legal_agent_step_runtime import build_legal_ollama_runtime_state

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_policy_compliance_flags_unlimited_liability() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-policy-1",
    )

    agent_state.clauses = [
        Clause(
            id="c_pol_1",
            text="The supplier accepts unlimited liability for all damages without cap.",
            category="liability",
            is_sensitive=True,
        ),
    ]

    await LegalPolicyComplianceStep().run_step(state=state, agent_state=agent_state)

    assert agent_state.policy_violations is not None
    assert len(agent_state.policy_violations) >= 1
