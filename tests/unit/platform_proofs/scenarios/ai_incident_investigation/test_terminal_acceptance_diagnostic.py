# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    TERMINAL_ACCEPTANCE_DIAGNOSTIC_PATH_ENV,
    TerminalAcceptanceDiagnostic,
    build_terminal_acceptance_diagnostic,
    derive_terminal_outcome,
    persist_terminal_acceptance_diagnostic,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)

pytestmark = pytest.mark.unit


def test_terminal_acceptance_diagnostic_exact_gate_values() -> None:
    diagnostic = build_terminal_acceptance_diagnostic(
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


def test_terminal_acceptance_diagnostic_does_not_change_outcome() -> None:
    diagnostic = build_terminal_acceptance_diagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=False,
        completion_mode=COMPLETION_UNRESOLVED,
        validation_errors=(),
        revision_pass=False,
        evidence_gathering_stop_reason="",
    )
    assert (
        derive_terminal_outcome(
            critic_verdict_passed=diagnostic.critic_verdict_passed,
            has_supported_diagnosis=diagnostic.has_supported_diagnosis,
            completion_mode=diagnostic.completion_mode,
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
    first = build_terminal_acceptance_diagnostic(**kwargs)
    second = build_terminal_acceptance_diagnostic(**kwargs)
    assert first == second


def test_terminal_acceptance_diagnostic_snapshots_are_independent() -> None:
    snapshot_a = build_terminal_acceptance_diagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        validation_errors=(),
        revision_pass=True,
        evidence_gathering_stop_reason="stop_a",
    )
    snapshot_b = build_terminal_acceptance_diagnostic(
        critic_verdict_passed=False,
        has_supported_diagnosis=False,
        completion_mode=COMPLETION_UNRESOLVED,
        validation_errors=("unsupported_inference:example",),
        revision_pass=False,
        evidence_gathering_stop_reason="stop_b",
    )
    assert snapshot_a is not snapshot_b
    assert snapshot_a != snapshot_b
    assert snapshot_a.evidence_gathering_stop_reason == "stop_a"
    assert snapshot_b.evidence_gathering_stop_reason == "stop_b"


def test_persist_terminal_acceptance_diagnostic_writes_json(tmp_path, monkeypatch) -> None:
    path = tmp_path / "diagnostic.json"
    monkeypatch.setenv(TERMINAL_ACCEPTANCE_DIAGNOSTIC_PATH_ENV, str(path))
    diagnostic = build_terminal_acceptance_diagnostic(
        critic_verdict_passed=True,
        has_supported_diagnosis=True,
        completion_mode=COMPLETION_SUPPORTED_DIAGNOSIS,
        validation_errors=("unsupported_inference:example",),
        revision_pass=True,
        evidence_gathering_stop_reason="planner_budget_exhausted",
    )
    persist_terminal_acceptance_diagnostic(diagnostic)
    assert json.loads(path.read_text(encoding="utf-8")) == diagnostic.to_json_dict()
