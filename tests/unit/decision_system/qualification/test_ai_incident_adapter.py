# © Artur Czarnecki. All rights reserved.

"""AI Incident qualification adapter integration tests (DS-E2E-14.3)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureCategory,
    DecisionFailureReason,
)
from testing_support.decision_e2e.failure_observation_adapter import (
    AI_INCIDENT_EPISTEMIC_FAILURE_ID,
    AI_INCIDENT_TOOL_USE_FAILURE_ID,
    observation_from_ai_incident_evaluation,
)


def test_adapter_maps_tool_runtime_not_exercised_to_model_tool_use_deficiency() -> None:
    observation = observation_from_ai_incident_evaluation(
        failures=(AI_INCIDENT_TOOL_USE_FAILURE_ID, "critic_falsification_missing"),
        evaluator_passed=False,
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.TOOL_USE_DEFICIENCY


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
