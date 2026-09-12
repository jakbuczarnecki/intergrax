# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Aggregated validation decision (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ValidationDecisionStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class SelfHealingValidationDecision:
    status: ValidationDecisionStatus
    confidence: float
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.explanation.strip():
            raise ValueError("explanation required")
        if not self.evidence_refs:
            raise ValueError("validation decision requires evidence_refs")
        if self.status is ValidationDecisionStatus.PASSED and self.failed_checks:
            raise ValueError("PASSED decision cannot list failed_checks")
        if self.status is ValidationDecisionStatus.FAILED and not self.failed_checks:
            raise ValueError("FAILED decision requires failed_checks")


__all__ = ["SelfHealingValidationDecision", "ValidationDecisionStatus"]
