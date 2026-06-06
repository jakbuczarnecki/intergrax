from __future__ import annotations

from intergrax.runtime.architecture.maturity_gate_evidence import (
    MaturityGateInputs,
    collect_harness_governance_signals,
    evaluate_maturity_gate_evidence,
)


def test_harness_governance_signals_pass_l3_and_l4() -> None:
    inputs = collect_harness_governance_signals()
    report = evaluate_maturity_gate_evidence(inputs)
    assert report.l3.passed is True
    assert report.l4_governance.passed is True
    assert report.l4_runtime.passed is True
    assert report.l4.passed is True


def test_l4_fails_when_adaptive_governance_fails() -> None:
    inputs = collect_harness_governance_signals()
    failing = inputs.model_copy(update={"adaptive_governance_passed": False})
    report = evaluate_maturity_gate_evidence(failing)
    assert report.l3.passed is True
    assert report.l4_governance.passed is False
    assert report.l4.passed is False


def test_l3_fails_when_compatibility_fails() -> None:
    inputs = MaturityGateInputs(
        capability_graph_compatible=False,
        metrics_pipeline_passed=True,
        architecture_debt_governance_passed=True,
        security_adversarial_passed=True,
        cost_governance_passed=True,
        evaluation_registry_available=True,
        multi_agent_acceptance_passed=True,
        adaptive_governance_passed=True,
        graph_rag_contract_valid=True,
        cost_forecast_available=True,
        cost_optimization_compliant=True,
    )
    report = evaluate_maturity_gate_evidence(inputs)
    assert report.l3.passed is False
