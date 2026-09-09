# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Qualification run result contract (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId
from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory


@dataclass(frozen=True, slots=True)
class DecisionQualificationRunResult:
    run_id: RunId
    classification: DecisionFailureClassification | None
    platform_contract_passed: bool
    model_behavior_passed: bool
    evaluator_passed: bool


def build_decision_qualification_run_result(
    *,
    run_id: RunId,
    observation: DecisionQualificationObservation,
    evaluator_passed: bool,
) -> DecisionQualificationRunResult:
    classification = classify_decision_failure(observation)
    platform_contract_passed = classification is None or not classification.is_platform_failure
    model_behavior_passed = classification is None or not classification.is_model_failure
    return DecisionQualificationRunResult(
        run_id=run_id,
        classification=classification,
        platform_contract_passed=platform_contract_passed,
        model_behavior_passed=model_behavior_passed,
        evaluator_passed=evaluator_passed,
    )


def category_from_run_result(
    result: DecisionQualificationRunResult,
) -> DecisionFailureCategory | None:
    if result.classification is None:
        return None
    return result.classification.category
