# © Artur Czarnecki. All rights reserved.

"""AI Incident qualification failure taxonomy coverage tests (DS-E2E-15J-T1)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.run_result import build_decision_qualification_run_result
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureOwner,
    DecisionFailureReason,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator_failure_vocabulary import (
    AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES,
    AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS,
    CITED_EVIDENCE_NOT_IN_GRAPH_PREFIX,
    GROUND_TRUTH_LEAK_PREFIX,
    TOOL_RUNTIME_NOT_EXERCISED,
    UNEXPECTED_OUTCOME_PREFIX,
    is_legal_ai_incident_failure_id,
)
from testing_support.decision_e2e.ai_incident_failure_qualification_mapping import (
    AI_INCIDENT_MAPPED_FAILURE_ID_PREFIXES,
    AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS,
    AiIncidentFailureMappingError,
    AiIncidentQualificationInputError,
    assert_ai_incident_failure_vocabulary_complete,
    mapped_ai_incident_failure_ids,
    qualification_spec_for_failure_id,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    observation_from_ai_incident_evaluation,
)
from testing_support.decision_e2e.reliability_qualification import (
    DecisionReliabilityQualificationRunRecord,
    validate_run_result_consistency,
)


def test_vocabulary_coverage_invariant() -> None:
    assert_ai_incident_failure_vocabulary_complete()
    assert AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS == mapped_ai_incident_failure_ids()
    assert not (AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS - AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS)
    assert not (AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS - AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS)
    assert AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES == AI_INCIDENT_MAPPED_FAILURE_ID_PREFIXES


@pytest.mark.parametrize("failure_id", sorted(AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS))
def test_every_legal_static_failure_classifies_deterministically(failure_id: str) -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(failure_id,),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    if failure_id == TOOL_RUNTIME_NOT_EXERCISED:
        assert result.category is DecisionFailureCategory.UNCLASSIFIED
        assert result.reason is DecisionFailureReason.UNCLASSIFIED
        return
    assert result.category is not DecisionFailureCategory.UNCLASSIFIED
    assert result.reason is not DecisionFailureReason.UNCLASSIFIED


@pytest.mark.parametrize(
    ("prefix", "sample"),
    (
        (CITED_EVIDENCE_NOT_IN_GRAPH_PREFIX, "evidence.telemetry.station_a"),
        (UNEXPECTED_OUTCOME_PREFIX, "RESOLVED"),
        (GROUND_TRUTH_LEAK_PREFIX, "oracle-marker"),
    ),
)
def test_parameterized_evaluator_failure_prefixes_classify(prefix: str, sample: str) -> None:
    failure_id = f"{prefix}{sample}"
    assert is_legal_ai_incident_failure_id(failure_id)
    result = classify_decision_failure(
        observation_from_ai_incident_evaluation(
            failures=(failure_id,),
            evaluator_passed=False,
        )
    )
    assert result is not None
    assert result.category is not DecisionFailureCategory.UNCLASSIFIED


def test_reverse_alignment_maps_to_epistemic_contradiction() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.EPISTEMIC_CONTRADICTION
    assert result.boundary is DecisionFailureBoundary.COMPLETION_RECONCILIATION
    assert result.owner is DecisionFailureOwner.MODEL


def test_forward_alignment_retains_epistemic_contradiction() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.EPISTEMIC_CONTRADICTION
    assert result.boundary is DecisionFailureBoundary.COMPLETION_RECONCILIATION
    assert result.owner is DecisionFailureOwner.MODEL


def test_unknown_failure_id_fails_closed() -> None:
    with pytest.raises(AiIncidentFailureMappingError):
        observation_from_ai_incident_evaluation(
            failures=("__unknown_failure_id__",),
            evaluator_passed=False,
        )


def test_empty_failures_with_evaluator_false_fail_closed() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category in {
        DecisionFailureCategory.OBSERVABILITY_GAP,
        DecisionFailureCategory.EVALUATOR_SEMANTICS,
        DecisionFailureCategory.UNCLASSIFIED,
    }


def test_evaluator_true_with_failures_is_contradictory() -> None:
    with pytest.raises(AiIncidentQualificationInputError):
        observation_from_ai_incident_evaluation(
            failures=("staffing_attendance_not_gathered",),
            evaluator_passed=True,
        )


def test_duplicate_failure_ids_do_not_change_classification() -> None:
    single = observation_from_ai_incident_evaluation(
        failures=("staffing_attendance_not_gathered",),
        evaluator_passed=False,
    )
    duplicated = observation_from_ai_incident_evaluation(
        failures=("staffing_attendance_not_gathered", "staffing_attendance_not_gathered"),
        evaluator_passed=False,
    )
    assert classify_decision_failure(single) == classify_decision_failure(duplicated)


def test_cross_family_precedence_prefers_insufficient_evidence() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(
            SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
            "staffing_attendance_not_gathered",
        ),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.reason is DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING


def test_order_independence_for_cross_family_failures() -> None:
    failures_a = (
        "follow_up_not_via_tools",
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    )
    failures_b = tuple(reversed(failures_a))
    result_a = classify_decision_failure(
        observation_from_ai_incident_evaluation(
            failures=failures_a,
            evaluator_passed=False,
        )
    )
    result_b = classify_decision_failure(
        observation_from_ai_incident_evaluation(
            failures=failures_b,
            evaluator_passed=False,
        )
    )
    assert result_a == result_b
    assert result_a is not None
    assert result_a.reason is DecisionFailureReason.PREMATURE_COMPLETION


def test_session_integrity_accepts_reverse_alignment_failure() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,),
        evaluator_passed=False,
    )
    run_result = build_decision_qualification_run_result(
        run_id=mint_run_id(),
        observation=observation,
        evaluator_passed=False,
    )
    record = DecisionReliabilityQualificationRunRecord(
        run_index=0,
        run_id=run_result.run_id,
        valid_model_trial=True,
        environment_event=False,
        completed=True,
        run_result=run_result,
        signals=None,
        block_reason=SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    )
    validate_run_result_consistency(record)
    assert run_result.classification is not None
    assert run_result.classification.category is not DecisionFailureCategory.UNCLASSIFIED


def test_session_integrity_still_aborts_on_unknown_failure_mapping_error() -> None:
    with pytest.raises(AiIncidentFailureMappingError):
        observation_from_ai_incident_evaluation(
            failures=("__totally_unknown_failure__",),
            evaluator_passed=False,
        )


def test_qualification_spec_exists_for_every_mapped_failure() -> None:
    for failure_id in sorted(AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS):
        spec = qualification_spec_for_failure_id(failure_id)
        assert spec.family is not None
