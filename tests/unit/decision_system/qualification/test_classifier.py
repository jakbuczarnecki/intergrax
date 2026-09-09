# © Artur Czarnecki. All rights reserved.

"""Classifier behavior tests for Decision qualification (DS-E2E-14.3)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationAmbiguityError,
)
from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.run_result import build_decision_qualification_run_result
from intergrax.decision_system.qualification.signals import (
    EnvironmentQualificationSignal,
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
    ProviderQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_for_credential_unavailable,
    observation_for_provider_rate_limit,
    observation_for_trace_not_finalized,
    observation_from_ai_incident_evaluation,
)


def _base_observation(**overrides: object) -> DecisionQualificationObservation:
    defaults = {
        "boundary": DecisionFailureBoundary.HOST_EXECUTION,
        "platform_contract": PlatformContractQualificationSignal(),
        "model_behavior": ModelBehaviorQualificationSignal(),
        "evaluator": EvaluatorQualificationSignal(passed=True),
        "provider": ProviderQualificationSignal(),
        "environment": EnvironmentQualificationSignal(),
        "observability": ObservabilityQualificationSignal(),
        "observability_complete": True,
    }
    defaults.update(overrides)
    return DecisionQualificationObservation(**defaults)  # type: ignore[arg-type]


def test_case_a_tool_runtime_not_exercised_is_not_tool_use_deficiency() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("tool_runtime_not_exercised",),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.UNCLASSIFIED
    assert result.reason is DecisionFailureReason.UNCLASSIFIED
    assert result.owner is DecisionFailureOwner.DECISION_SYSTEM
    assert result.diagnostic_code is DecisionFailureDiagnosticCode.UNCLASSIFIED


def test_case_b_epistemic_contradiction_platform_enforcement_pass() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("unsupported_inference:unresolved_with_supported_diagnosis",),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.EPISTEMIC_CONTRADICTION
    assert result.boundary is DecisionFailureBoundary.COMPLETION_RECONCILIATION
    assert not result.is_platform_failure


def test_case_c_trace_not_finalized() -> None:
    observation = observation_for_trace_not_finalized()
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.PLATFORM_CONTRACT
    assert result.reason is DecisionFailureReason.TRACE_RUN_NOT_FINALIZED
    assert result.owner is DecisionFailureOwner.EXECUTION_ENGINE


def test_case_d_provider_rate_limit() -> None:
    observation = observation_for_provider_rate_limit()
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
    assert result.reason is DecisionFailureReason.RATE_LIMIT
    assert result.retryability is DecisionRetryability.RETRIABLE


def test_case_e_credential_unavailable() -> None:
    observation = observation_for_credential_unavailable()
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.ENVIRONMENT
    assert result.reason is DecisionFailureReason.CREDENTIAL_UNAVAILABLE


def test_pass_run_returns_none_classification() -> None:
    observation = _base_observation()
    assert classify_decision_failure(observation) is None


def test_evaluator_fail_without_semantics_error_is_not_evaluator_owner() -> None:
    observation = _base_observation(
        model_behavior=ModelBehaviorQualificationSignal(
            tool_use_deficiency=True,
            behavior_boundary=DecisionFailureBoundary.TOOL_DISPATCH,
        ),
        evaluator=EvaluatorQualificationSignal(passed=False),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.owner is DecisionFailureOwner.MODEL


def test_observability_gap_when_incomplete() -> None:
    observation = _base_observation(
        observability_complete=False,
        observability=ObservabilityQualificationSignal(critical_boundary_unknown=True),
        evaluator=EvaluatorQualificationSignal(passed=False),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.OBSERVABILITY_GAP


def test_unclassified_fallback_when_only_evaluator_fail() -> None:
    observation = _base_observation(evaluator=EvaluatorQualificationSignal(passed=False))
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.UNCLASSIFIED


def test_classifier_determinism() -> None:
    observation = observation_for_provider_rate_limit()
    first = classify_decision_failure(observation)
    second = classify_decision_failure(observation)
    assert first == second


def test_first_failure_boundary_prefers_earlier_causal_boundary() -> None:
    observation = _base_observation(
        boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
        model_behavior=ModelBehaviorQualificationSignal(
            epistemic_contradiction=True,
            behavior_boundary=DecisionFailureBoundary.REASONING,
        ),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.boundary is DecisionFailureBoundary.REASONING


def test_duplicate_match_protection_raises_ambiguity() -> None:
    observation = _base_observation(
        environment=EnvironmentQualificationSignal(
            credential_unavailable=True,
            qualification_disabled=True,
        )
    )
    with pytest.raises(DecisionFailureClassificationAmbiguityError):
        classify_decision_failure(observation)


def test_platform_enforcement_success_run_result_axes() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("unsupported_inference:unresolved_with_supported_diagnosis",),
        evaluator_passed=False,
    )
    run_result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation,
        evaluator_passed=False,
    )
    assert run_result.platform_contract_passed
    assert not run_result.model_behavior_passed
