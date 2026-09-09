# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Qualification run result contract (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId
from intergrax.decision_system.qualification.axis_outcome import (
    DecisionQualificationAxisOutcome,
    derive_axis_outcomes,
)
from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory


class DecisionQualificationRunResultConsistencyError(ValueError):
    """Fail-closed when axis outcomes contradict classification facts."""


@dataclass(frozen=True, slots=True)
class DecisionQualificationRunResult:
    run_id: RunId
    classification: DecisionFailureClassification | None
    platform_outcome: DecisionQualificationAxisOutcome
    model_outcome: DecisionQualificationAxisOutcome
    evaluator_outcome: DecisionQualificationAxisOutcome
    evaluator_passed: bool

    @property
    def platform_contract_passed(self) -> bool:
        return self.platform_outcome is DecisionQualificationAxisOutcome.PASS

    @property
    def model_behavior_passed(self) -> bool:
        return self.model_outcome is DecisionQualificationAxisOutcome.PASS


def validate_run_result_axis_consistency(
    result: DecisionQualificationRunResult,
) -> None:
    classification = result.classification
    if classification is None:
        return
    if (
        classification.category is DecisionFailureCategory.MODEL_BEHAVIOR
        and result.model_outcome is not DecisionQualificationAxisOutcome.FAIL
    ):
        raise DecisionQualificationRunResultConsistencyError(
            "model failure classification requires model_outcome=FAIL"
        )
    if (
        classification.category is DecisionFailureCategory.PLATFORM_CONTRACT
        and result.platform_outcome is not DecisionQualificationAxisOutcome.FAIL
    ):
        raise DecisionQualificationRunResultConsistencyError(
            "platform failure classification requires platform_outcome=FAIL"
        )
    if (
        classification.category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE
        and result.model_outcome is DecisionQualificationAxisOutcome.PASS
    ):
        raise DecisionQualificationRunResultConsistencyError(
            "provider infrastructure failure cannot produce model_outcome=PASS"
        )
    if (
        classification.category is DecisionFailureCategory.ENVIRONMENT
        and result.model_outcome is DecisionQualificationAxisOutcome.PASS
    ):
        raise DecisionQualificationRunResultConsistencyError(
            "environment failure cannot produce model_outcome=PASS"
        )


def build_decision_qualification_run_result(
    *,
    run_id: RunId,
    observation: DecisionQualificationObservation,
    evaluator_passed: bool,
) -> DecisionQualificationRunResult:
    classification = classify_decision_failure(observation)
    platform_outcome, model_outcome, evaluator_outcome = derive_axis_outcomes(
        observation=observation,
        classification=classification,
        evaluator_passed=evaluator_passed,
    )
    result = DecisionQualificationRunResult(
        run_id=run_id,
        classification=classification,
        platform_outcome=platform_outcome,
        model_outcome=model_outcome,
        evaluator_outcome=evaluator_outcome,
        evaluator_passed=evaluator_passed,
    )
    validate_run_result_axis_consistency(result)
    return result


def category_from_run_result(
    result: DecisionQualificationRunResult,
) -> DecisionFailureCategory | None:
    if result.classification is None:
        return None
    return result.classification.category
