# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Integration test: LegalDecisionEnforcementStep (deterministic rules; real Ollama runtime).

The step does not call the LLM, but the test uses the same runtime wiring as other
legal step tests. If Ollama is unreachable, the test is skipped.
"""

from __future__ import annotations

import pytest

from legal.tests.support.step_runtime import build_legal_ollama_runtime_state
from testing_support.builder import require_ollama_reachable

from legal.domain.legal_agent_state import (
    LegalCheck,
    LegalDecision,
    PolicyViolation,
)
from legal.steps.legal_decision_enforcement_step import (
    LegalDecisionEnforcementStep,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_enforcement_escalates_approve_to_conditional_on_policy_violation() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-enf-policy",
    )

    agent_state.decision = LegalDecision(
        status="APPROVE",
        confidence=0.95,
        blocking_issues=[],
        summary="Looks fine.",
    )
    agent_state.legal_checks = [
        LegalCheck(clause_id="c1", valid=True, source="LOW", details="ok"),
    ]
    agent_state.policy_violations = [
        PolicyViolation(
            clause_id="c1",
            policy_rule="no_auto_renew",
            violation="Auto-renew present",
            suggested_fix="Remove auto-renew",
            severity="MEDIUM",
        ),
    ]

    await LegalDecisionEnforcementStep().run_step(state=state, agent_state=agent_state)

    assert agent_state.decision.status == "CONDITIONAL"
    assert agent_state.decision_enforcement_modified is True
    assert any("Enforcement:" in str(x) for x in agent_state.decision.blocking_issues)


@pytest.mark.asyncio
async def test_enforcement_rejects_on_failed_legal_check() -> None:
    require_ollama_reachable()

    state, agent_state = build_legal_ollama_runtime_state(
        run_id="run-legal-enf-risk",
    )

    agent_state.decision = LegalDecision(
        status="APPROVE",
        confidence=0.9,
        blocking_issues=[],
        summary="OK",
    )
    agent_state.legal_checks = [
        LegalCheck(
            clause_id="c1",
            valid=False,
            source="HIGH",
            details="Blocking risk.",
        ),
    ]
    agent_state.policy_violations = []

    await LegalDecisionEnforcementStep().run_step(state=state, agent_state=agent_state)

    assert agent_state.decision.status == "REJECT"
    assert agent_state.decision_enforcement_modified is True
