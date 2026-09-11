# © Artur Czarnecki. All rights reserved.

"""Qualification-owned failure classification boundary (no reflection fallbacks)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


class ClassificationParseError(ValueError):
    """Malformed persisted classification payload."""


@dataclass(frozen=True, slots=True)
class QualificationFailureView:
    category: DecisionFailureCategory
    reason: DecisionFailureReason
    boundary: DecisionFailureBoundary
    owner: DecisionFailureOwner

    @property
    def is_platform_failure(self) -> bool:
        return self.category is DecisionFailureCategory.PLATFORM_CONTRACT

    @property
    def is_model_failure(self) -> bool:
        return self.category is DecisionFailureCategory.MODEL_BEHAVIOR


def failure_view_from_classification(
    classification: DecisionFailureClassification,
) -> QualificationFailureView:
    return QualificationFailureView(
        category=classification.category,
        reason=classification.reason,
        boundary=classification.boundary,
        owner=classification.owner,
    )


def classification_from_persisted_dict(
    payload: dict[str, object],
) -> DecisionFailureClassification:
    try:
        return DecisionFailureClassification(
            category=DecisionFailureCategory(str(payload["category"])),
            reason=DecisionFailureReason(str(payload["reason"])),
            boundary=DecisionFailureBoundary(str(payload["boundary"])),
            owner=DecisionFailureOwner(str(payload["owner"])),
            retryability=DecisionRetryability(str(payload["retryability"])),
            diagnostic_code=DecisionFailureDiagnosticCode(str(payload["diagnostic_code"])),
        )
    except (KeyError, ValueError) as exc:
        raise ClassificationParseError(f"invalid classification payload: {exc}") from exc
