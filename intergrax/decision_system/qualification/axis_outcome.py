# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Axis outcome semantics for Decision qualification reliability (DS-E2E-14.3c)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    boundary_rank,
)


class DecisionQualificationAxisOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUABLE = "not_evaluable"


_MODEL_REACHED_BOUNDARY = DecisionFailureBoundary.HOST_EXECUTION
_EVALUATOR_REACHED_BOUNDARY = DecisionFailureBoundary.TERMINAL_ACCEPTANCE


def _platform_axis_evaluable(
    observation: DecisionQualificationObservation,
    classification: DecisionFailureClassification | None,
) -> bool:
    if not observation.observability_complete:
        return False
    if classification is None:
        return True
    category = classification.category
    if category is DecisionFailureCategory.PLATFORM_CONTRACT:
        return True
    if category in (
        DecisionFailureCategory.MODEL_BEHAVIOR,
        DecisionFailureCategory.EVALUATOR_SEMANTICS,
    ):
        return True
    if category is DecisionFailureCategory.ENVIRONMENT:
        return False
    if category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE:
        return boundary_rank(classification.boundary) > boundary_rank(
            DecisionFailureBoundary.PROVIDER_BINDING
        )
    if category in (
        DecisionFailureCategory.OBSERVABILITY_GAP,
        DecisionFailureCategory.UNCLASSIFIED,
    ):
        return False
    return False


def _model_axis_evaluable(
    observation: DecisionQualificationObservation,
    classification: DecisionFailureClassification | None,
) -> bool:
    if not observation.observability_complete:
        return False
    if classification is None:
        return True
    category = classification.category
    if category is DecisionFailureCategory.MODEL_BEHAVIOR:
        return True
    if category in (
        DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
        DecisionFailureCategory.ENVIRONMENT,
    ):
        return False
    if category is DecisionFailureCategory.PLATFORM_CONTRACT:
        return boundary_rank(classification.boundary) >= boundary_rank(_MODEL_REACHED_BOUNDARY)
    if category is DecisionFailureCategory.EVALUATOR_SEMANTICS:
        return True
    if category in (
        DecisionFailureCategory.OBSERVABILITY_GAP,
        DecisionFailureCategory.UNCLASSIFIED,
    ):
        return False
    return False


def _evaluator_axis_evaluable(
    observation: DecisionQualificationObservation,
    classification: DecisionFailureClassification | None,
) -> bool:
    if not observation.observability_complete:
        return False
    if classification is None:
        return True
    category = classification.category
    if category in (
        DecisionFailureCategory.ENVIRONMENT,
        DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
    ):
        return False
    if category in (
        DecisionFailureCategory.MODEL_BEHAVIOR,
        DecisionFailureCategory.EVALUATOR_SEMANTICS,
    ):
        return True
    if category is DecisionFailureCategory.PLATFORM_CONTRACT:
        return boundary_rank(classification.boundary) >= boundary_rank(
            _EVALUATOR_REACHED_BOUNDARY
        )
    if category in (
        DecisionFailureCategory.OBSERVABILITY_GAP,
        DecisionFailureCategory.UNCLASSIFIED,
    ):
        return False
    return False


def _platform_axis_failure(classification: DecisionFailureClassification | None) -> bool:
    return (
        classification is not None
        and classification.category is DecisionFailureCategory.PLATFORM_CONTRACT
    )


def _model_axis_failure(classification: DecisionFailureClassification | None) -> bool:
    return (
        classification is not None
        and classification.category is DecisionFailureCategory.MODEL_BEHAVIOR
    )


def derive_axis_outcomes(
    *,
    observation: DecisionQualificationObservation,
    classification: DecisionFailureClassification | None,
    evaluator_passed: bool,
) -> tuple[
    DecisionQualificationAxisOutcome,
    DecisionQualificationAxisOutcome,
    DecisionQualificationAxisOutcome,
]:
    """Pure derivation: observation/classification facts to typed axis outcomes."""
    if _platform_axis_evaluable(observation, classification):
        platform_outcome = (
            DecisionQualificationAxisOutcome.FAIL
            if _platform_axis_failure(classification)
            else DecisionQualificationAxisOutcome.PASS
        )
    else:
        platform_outcome = DecisionQualificationAxisOutcome.NOT_EVALUABLE

    if _model_axis_evaluable(observation, classification):
        model_outcome = (
            DecisionQualificationAxisOutcome.FAIL
            if _model_axis_failure(classification)
            else DecisionQualificationAxisOutcome.PASS
        )
    else:
        model_outcome = DecisionQualificationAxisOutcome.NOT_EVALUABLE

    if _evaluator_axis_evaluable(observation, classification):
        evaluator_outcome = (
            DecisionQualificationAxisOutcome.PASS
            if evaluator_passed
            else DecisionQualificationAxisOutcome.FAIL
        )
    else:
        evaluator_outcome = DecisionQualificationAxisOutcome.NOT_EVALUABLE

    return platform_outcome, model_outcome, evaluator_outcome
