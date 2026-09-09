# © Artur Czarnecki. All rights reserved.

"""AI Incident qualification adapter integration tests (DS-E2E-14.3 / DS-E2E-15B.1)."""

from __future__ import annotations

import pytest

from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureCategory,
    DecisionFailureReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
    CompletionReconciliationDiagnostic,
    CompletionReconciliationError,
    CompletionReconciliationFailureReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    CompletionIntent,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID,
    AI_INCIDENT_EPISTEMIC_FAILURE_ID,
    observation_from_ai_incident_evaluation,
    observation_from_scenario_execution_exception,
)


def test_adapter_does_not_map_tool_runtime_not_exercised_to_tool_use_deficiency() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID,),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.reason is not DecisionFailureReason.TOOL_USE_DEFICIENCY
    assert result.category is DecisionFailureCategory.UNCLASSIFIED


def test_adapter_maps_insufficient_evidence_gathering_failure_ids() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("staffing_attendance_not_gathered", "follow_up_not_via_tools"),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING


def test_adapter_precedence_prefers_missing_attendance_over_follow_up_not_via_tools() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("follow_up_not_via_tools", "staffing_attendance_not_gathered"),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.reason is DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING


def test_adapter_maps_revision_flow_failures_to_premature_completion() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(
            "telemetry_visible_before_revision",
            "critic_falsification_missing",
            "evidence_challenge_missing",
        ),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.PREMATURE_COMPLETION


def test_adapter_maps_follow_up_not_via_tools_to_premature_completion_when_isolated() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=("follow_up_not_via_tools",),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.reason is DecisionFailureReason.PREMATURE_COMPLETION


def test_adapter_maps_epistemic_failure_id_without_string_classifier() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(AI_INCIDENT_EPISTEMIC_FAILURE_ID,),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.EPISTEMIC_CONTRADICTION
    assert not result.is_platform_failure


def test_adapter_maps_completion_reconciliation_error_to_model_unsupported_completion() -> None:
    observation = observation_from_scenario_execution_exception(
        CompletionReconciliationError(
            CompletionReconciliationFailureReason.VALIDATION_ERRORS_PRESENT,
            diagnostic=CompletionReconciliationDiagnostic(
                model_intent=CompletionIntent.SUPPORTED_DIAGNOSIS,
                critic_verdict_passed=True,
                has_supported_diagnosis=True,
                validation_errors=("some_error",),
                evidence_gathering_stop_reason="planner_final_answer",
            ),
        )
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.UNSUPPORTED_COMPLETION


@pytest.mark.parametrize(
    ("failures", "expected_reason"),
    (
        (("tool_runtime_not_exercised",), DecisionFailureReason.UNCLASSIFIED),
        (("follow_up_not_via_tools",), DecisionFailureReason.PREMATURE_COMPLETION),
        (("comparison_evidence_not_gathered",), DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING),
        (("staffing_attendance_not_gathered",), DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING),
        (("telemetry_evidence_not_in_graph",), DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING),
        (
            (
                "telemetry_visible_before_revision",
                "critic_falsification_missing",
            ),
            DecisionFailureReason.PREMATURE_COMPLETION,
        ),
        ((AI_INCIDENT_EPISTEMIC_FAILURE_ID,), DecisionFailureReason.EPISTEMIC_CONTRADICTION),
    ),
)
def test_adapter_golden_mapping_matrix(
    failures: tuple[str, ...],
    expected_reason: DecisionFailureReason,
) -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=failures,
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR or (
        expected_reason is DecisionFailureReason.UNCLASSIFIED
        and result.category is DecisionFailureCategory.UNCLASSIFIED
    )
    assert result.reason is expected_reason
