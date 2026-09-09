# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Reusable typed contracts for Decision failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassification,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
    earliest_boundary,
)


class DecisionFailurePredicate(Protocol):
    def __call__(
        self,
        observation: DecisionQualificationObservation,
    ) -> bool:
        ...


class DecisionFailureBoundaryResolver(Protocol):
    def __call__(
        self,
        observation: DecisionQualificationObservation,
        default_boundary: DecisionFailureBoundary,
    ) -> DecisionFailureBoundary:
        ...


def resolve_default_boundary(
    observation: DecisionQualificationObservation,
    default_boundary: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    return earliest_boundary(observation.boundary, default_boundary)


def resolve_model_behavior_boundary(
    observation: DecisionQualificationObservation,
    default_boundary: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    explicit = observation.model_behavior.behavior_boundary
    if explicit is not None:
        return explicit
    return earliest_boundary(observation.boundary, default_boundary)


def resolve_platform_violation_boundary(
    observation: DecisionQualificationObservation,
    default_boundary: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    explicit = observation.platform_contract.violation_boundary
    if explicit is not None:
        return explicit
    return earliest_boundary(observation.boundary, default_boundary)


def build_classification(
    *,
    category: DecisionFailureCategory,
    reason: DecisionFailureReason,
    boundary: DecisionFailureBoundary,
    owner: DecisionFailureOwner,
    retryability: DecisionRetryability,
    diagnostic_code: DecisionFailureDiagnosticCode,
) -> DecisionFailureClassification:
    return DecisionFailureClassification(
        category=category,
        reason=reason,
        boundary=boundary,
        owner=owner,
        retryability=retryability,
        diagnostic_code=diagnostic_code,
    )


@dataclass(frozen=True, slots=True)
class DecisionFailureRuleSpec:
    category: DecisionFailureCategory
    reason: DecisionFailureReason
    owner: DecisionFailureOwner
    retryability: DecisionRetryability
    diagnostic_code: DecisionFailureDiagnosticCode
    default_boundary: DecisionFailureBoundary
    predicate: DecisionFailurePredicate
    boundary_resolver: DecisionFailureBoundaryResolver


@dataclass(frozen=True, slots=True)
class SpecDecisionFailureRule:
    spec: DecisionFailureRuleSpec

    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not self.spec.predicate(observation):
            return None
        return build_classification(
            category=self.spec.category,
            reason=self.spec.reason,
            boundary=self.spec.boundary_resolver(
                observation,
                self.spec.default_boundary,
            ),
            owner=self.spec.owner,
            retryability=self.spec.retryability,
            diagnostic_code=self.spec.diagnostic_code,
        )


def rule_from_spec(spec: DecisionFailureRuleSpec) -> SpecDecisionFailureRule:
    return SpecDecisionFailureRule(spec=spec)
