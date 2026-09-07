# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    TerminalAcceptanceDiagnostic,
    capture_terminal_acceptance_diagnostic,
    derive_terminal_outcome,
    peek_last_terminal_acceptance_diagnostic,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)

pytestmark = pytest.mark.unit


def test_terminal_acceptance_diagnostic_exact_gate_values() -> None:
    diagnostic = capture_terminal_acceptance_diagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        validation_errors=("unsupported_inference:example",),
        revision_pass=True,
        evidence_gathering_stop_reason="planner_budget_exhausted",
    )
    assert diagnostic == TerminalAcceptanceDiagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        validation_errors=("unsupported_inference:example",),
        revision_pass=True,
        evidence_gathering_stop_reason="planner_budget_exhausted",
    )
    assert peek_last_terminal_acceptance_diagnostic() == diagnostic


def test_terminal_acceptance_diagnostic_does_not_change_outcome() -> None:
    capture_terminal_acceptance_diagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=False,
        completion_mode=COMPLETION_UNRESOLVED,
        validation_errors=(),
        revision_pass=False,
        evidence_gathering_stop_reason="",
    )
    assert (
        derive_terminal_outcome(
            critic_verdict_passed=True,
            has_supported_diagnosis=False,
            completion_mode=COMPLETION_UNRESOLVED,
        )
        == OUTCOME_UNRESOLVED
    )
    assert (
        derive_terminal_outcome(
            critic_verdict_passed=True,
            has_supported_diagnosis=True,
            completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        )
        == OUTCOME_RESOLVED
    )


def test_terminal_acceptance_diagnostic_is_deterministic() -> None:
    kwargs = {
        "critic_verdict_passed": False,
        "has_supported_diagnosis": True,
        "completion_mode": COMPLETION_SUPPORTED_DIAGNOSIS,
        "validation_errors": ("unsupported_inference:h1_only_causal_diagnosis_insufficient",),
        "revision_pass": False,
        "evidence_gathering_stop_reason": "critic_follow_up_complete",
    }
    first = capture_terminal_acceptance_diagnostic(**kwargs)
    second = capture_terminal_acceptance_diagnostic(**kwargs)
    assert first == second
