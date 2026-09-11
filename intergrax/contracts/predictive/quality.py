# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Central predictive quality assessment (PREDICTIVE R4 governance)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive.completeness import PredictiveContextCompleteness


@dataclass(frozen=True, slots=True)
class PredictiveQualityDimension:
    """Single bounded quality factor in [0.0, 1.0]."""

    label: str
    score: float
    rationale: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class PredictiveQualityAssessment:
    """
    Explains governed confidence — never a raw scalar without factors.

    Governed confidence ≈ context × analyzer × evidence (see runtime governance).
    """

    context_quality: PredictiveQualityDimension
    analyzer_quality: PredictiveQualityDimension
    evidence_quality: PredictiveQualityDimension
    confidence_quality: PredictiveQualityDimension
    completeness: PredictiveContextCompleteness
    governed_confidence: float
    explanation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 <= self.governed_confidence <= 1.0):
            raise ValueError("governed_confidence must be in [0.0, 1.0]")


__all__ = [
    "PredictiveQualityAssessment",
    "PredictiveQualityDimension",
]
