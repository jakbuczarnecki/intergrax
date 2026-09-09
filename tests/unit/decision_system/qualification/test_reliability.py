# © Artur Czarnecki. All rights reserved.

"""Reliability aggregation tests for Decision qualification (DS-E2E-14.3)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.decision_system.qualification.axis_outcome import DecisionQualificationAxisOutcome
from intergrax.decision_system.qualification.reliability import aggregate_decision_reliability
from intergrax.decision_system.qualification.run_result import build_decision_qualification_run_result
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureCategory,
    DecisionFailureReason,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_for_provider_rate_limit,
    observation_for_trace_not_finalized,
    observation_from_ai_incident_evaluation,
)


def _run_from_ai_incident(*failures: str):
    observation = observation_from_ai_incident_evaluation(
        failures=failures,
        evaluator_passed=False,
    )
    return build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation,
        evaluator_passed=False,
    )


def test_ds_e2e_14_1b_five_run_replay() -> None:
    run_results = (
        _run_from_ai_incident("staffing_attendance_not_gathered"),
        _run_from_ai_incident("staffing_attendance_not_gathered"),
        _run_from_ai_incident("staffing_attendance_not_gathered"),
        _run_from_ai_incident("unsupported_inference:unresolved_with_supported_diagnosis"),
        _run_from_ai_incident("unsupported_inference:unresolved_with_supported_diagnosis"),
    )
    summary = aggregate_decision_reliability(run_results)

    assert summary.total_runs == 5
    assert summary.platform_evaluable_count == 5
    assert summary.platform_pass_count == 5
    assert summary.platform_failure_count == 0
    assert summary.model_evaluable_count == 5
    assert summary.model_failure_count == 5
    assert summary.model_pass_count == 0
    assert summary.platform_reliability == 1.0
    assert summary.model_reliability == 0.0


def test_reliability_aggregation_five_model_failures() -> None:
    run_results = tuple(
        _run_from_ai_incident("staffing_attendance_not_gathered") for _ in range(3)
    ) + tuple(
        _run_from_ai_incident("unsupported_inference:unresolved_with_supported_diagnosis")
        for _ in range(2)
    )
    summary = aggregate_decision_reliability(run_results)
    assert summary.platform_pass_count == 5
    assert summary.model_failure_count == 5
    assert summary.model_pass_count == 0


def test_mixed_failure_counts_stay_separated() -> None:
    model_pass = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_from_ai_incident_evaluation(
            failures=(),
            evaluator_passed=True,
        ),
        evaluator_passed=True,
    )
    model_tool_fail = _run_from_ai_incident("staffing_attendance_not_gathered")
    model_epistemic_fail = _run_from_ai_incident(
        "unsupported_inference:unresolved_with_supported_diagnosis"
    )
    provider_fail = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_for_provider_rate_limit(),
        evaluator_passed=False,
    )
    platform_fail = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_for_trace_not_finalized(),
        evaluator_passed=False,
    )

    run_results = (
        model_pass,
        model_pass,
        model_pass,
        model_pass,
        model_pass,
        model_pass,
        model_tool_fail,
        model_epistemic_fail,
        provider_fail,
        platform_fail,
    )
    summary = aggregate_decision_reliability(run_results)

    assert summary.total_runs == 10
    assert summary.model_pass_count == 7
    assert summary.model_failure_count == 2
    assert summary.model_evaluable_count == 9
    assert summary.model_not_evaluable_count == 1
    assert summary.platform_pass_count == 8
    assert summary.platform_failure_count == 1
    assert summary.platform_evaluable_count == 9
    assert summary.provider_infra_failure_count == 1
    assert summary.environment_failure_count == 0
    assert provider_fail.classification is not None
    assert provider_fail.classification.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert provider_fail.model_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE
    assert platform_fail.classification is not None
    assert platform_fail.classification.category is DecisionFailureCategory.PLATFORM_CONTRACT


def test_zero_division_returns_none_rates() -> None:
    summary = aggregate_decision_reliability(())
    assert summary.total_runs == 0
    assert summary.platform_reliability is None
    assert summary.model_reliability is None
    assert summary.evaluator_pass_rate is None
    assert summary.platform_evaluation_coverage is None
    assert summary.model_evaluation_coverage is None


def test_model_failure_reasons_are_counted_via_run_axes() -> None:
    evidence_fail = _run_from_ai_incident("staffing_attendance_not_gathered")
    epistemic_fail = _run_from_ai_incident(
        "unsupported_inference:unresolved_with_supported_diagnosis"
    )
    assert evidence_fail.classification is not None
    assert evidence_fail.classification.reason is DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING
    assert epistemic_fail.classification is not None
    assert epistemic_fail.classification.reason is DecisionFailureReason.EPISTEMIC_CONTRADICTION
