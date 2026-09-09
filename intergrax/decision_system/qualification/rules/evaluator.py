# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Evaluator semantics failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules.contracts import (
    DecisionFailureRuleSpec,
    resolve_default_boundary,
    rule_from_spec,
)
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


def _false_negative(observation: DecisionQualificationObservation) -> bool:
    return observation.evaluator.false_negative


def _false_positive(observation: DecisionQualificationObservation) -> bool:
    return observation.evaluator.false_positive


def _criterion_semantics_invalid(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.evaluator.criterion_semantics_invalid


def _contract_error(observation: DecisionQualificationObservation) -> bool:
    return observation.evaluator.contract_error


_EVALUATOR_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
        reason=DecisionFailureReason.FALSE_NEGATIVE,
        owner=DecisionFailureOwner.EVALUATOR,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_FALSE_NEGATIVE,
        default_boundary=DecisionFailureBoundary.EVALUATOR,
        predicate=_false_negative,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
        reason=DecisionFailureReason.FALSE_POSITIVE,
        owner=DecisionFailureOwner.EVALUATOR,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_FALSE_POSITIVE,
        default_boundary=DecisionFailureBoundary.EVALUATOR,
        predicate=_false_positive,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
        reason=DecisionFailureReason.CRITERION_SEMANTICS_INVALID,
        owner=DecisionFailureOwner.EVALUATOR,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_CRITERION_SEMANTICS,
        default_boundary=DecisionFailureBoundary.EVALUATOR,
        predicate=_criterion_semantics_invalid,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
        reason=DecisionFailureReason.EVALUATOR_CONTRACT_ERROR,
        owner=DecisionFailureOwner.EVALUATOR,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_CONTRACT_ERROR,
        default_boundary=DecisionFailureBoundary.EVALUATOR,
        predicate=_contract_error,
        boundary_resolver=resolve_default_boundary,
    ),
)

EVALUATOR_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _EVALUATOR_RULE_SPECS
)
