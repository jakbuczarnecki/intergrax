# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Failure classification result and rule protocol (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


class DecisionFailureClassificationAmbiguityError(RuntimeError):
    """Raised when multiple rules claim incompatible root causes at one boundary."""


@dataclass(frozen=True, slots=True)
class DecisionFailureClassification:
    category: DecisionFailureCategory
    reason: DecisionFailureReason
    boundary: DecisionFailureBoundary
    owner: DecisionFailureOwner
    retryability: DecisionRetryability
    diagnostic_code: DecisionFailureDiagnosticCode

    @property
    def is_platform_failure(self) -> bool:
        return self.category is DecisionFailureCategory.PLATFORM_CONTRACT

    @property
    def is_model_failure(self) -> bool:
        return self.category is DecisionFailureCategory.MODEL_BEHAVIOR

    @property
    def is_retriable(self) -> bool:
        return self.retryability is DecisionRetryability.RETRIABLE


class DecisionFailureClassificationRule(Protocol):
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        ...
