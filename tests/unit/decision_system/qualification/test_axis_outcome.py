# © Artur Czarnecki. All rights reserved.

"""Axis outcome derivation tests (DS-E2E-14.3c)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.decision_system.qualification.axis_outcome import DecisionQualificationAxisOutcome
from intergrax.decision_system.qualification.reliability import aggregate_decision_reliability
from intergrax.decision_system.qualification.run_result import (
    DecisionQualificationRunResult,
    build_decision_qualification_run_result,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_for_credential_unavailable,
    observation_for_provider_rate_limit,
    observation_for_trace_not_finalized,
    observation_from_ai_incident_evaluation,
)


def _success_run() -> DecisionQualificationRunResult:
    return build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_from_ai_incident_evaluation(
            failures=(),
            evaluator_passed=True,
        ),
        evaluator_passed=True,
    )


def _model_fail_run() -> DecisionQualificationRunResult:
    return build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_from_ai_incident_evaluation(
            failures=("tool_runtime_not_exercised",),
            evaluator_passed=False,
        ),
        evaluator_passed=False,
    )


def test_model_success_classification_none() -> None:
    result = _success_run()
    assert result.classification is None
    assert result.model_outcome is DecisionQualificationAxisOutcome.PASS
    assert result.platform_outcome is DecisionQualificationAxisOutcome.PASS
    assert result.evaluator_outcome is DecisionQualificationAxisOutcome.PASS


def test_model_behavior_failure() -> None:
    result = _model_fail_run()
    assert result.classification is not None
    assert result.classification.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.model_outcome is DecisionQualificationAxisOutcome.FAIL
    assert result.platform_outcome is DecisionQualificationAxisOutcome.PASS


def test_provider_failure_model_not_evaluable() -> None:
    result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_for_provider_rate_limit(),
        evaluator_passed=False,
    )
    assert result.classification is not None
    assert result.classification.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.model_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE
    assert result.platform_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE
    assert result.evaluator_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE


def test_environment_failure_axes_not_evaluable() -> None:
    result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_for_credential_unavailable(),
        evaluator_passed=False,
    )
    assert result.classification is not None
    assert result.classification.category is DecisionFailureCategory.ENVIRONMENT
    assert result.model_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE
    assert result.platform_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE
    assert result.evaluator_outcome is DecisionQualificationAxisOutcome.NOT_EVALUABLE


def test_platform_contract_failure() -> None:
    result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation_for_trace_not_finalized(),
        evaluator_passed=False,
    )
    assert result.classification is not None
    assert result.classification.category is DecisionFailureCategory.PLATFORM_CONTRACT
    assert result.platform_outcome is DecisionQualificationAxisOutcome.FAIL


def test_ds_e2e_14_3b_twenty_run_replay() -> None:
    run_results = tuple(_model_fail_run() for _ in range(16)) + tuple(
        _success_run() for _ in range(3)
    ) + (
        build_decision_qualification_run_result(
            run_id=mint_run_id(),
            observation=observation_for_provider_rate_limit(),
            evaluator_passed=False,
        ),
    )
    summary = aggregate_decision_reliability(run_results)

    assert summary.total_runs == 20
    assert summary.model_evaluable_count == 19
    assert summary.model_pass_count == 3
    assert summary.model_failure_count == 16
    assert summary.model_not_evaluable_count == 1
    assert summary.model_reliability == 3 / 19
    assert summary.model_evaluation_coverage == 19 / 20
    assert summary.provider_infra_failure_count == 1
