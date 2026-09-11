# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Context quality read model for predictive governance (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive.completeness import PredictiveContextCompleteness


@dataclass(frozen=True, slots=True)
class PredictiveContextQualityReport:
    """Operator-auditable context quality — not diagnostic authority."""

    completeness: PredictiveContextCompleteness
    coverage: float
    freshness_score: float
    reliability: float
    missing_sections: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0.0, 1.0]")
        if not (0.0 <= self.freshness_score <= 1.0):
            raise ValueError("freshness_score must be in [0.0, 1.0]")
        if not (0.0 <= self.reliability <= 1.0):
            raise ValueError("reliability must be in [0.0, 1.0]")


__all__ = ["PredictiveContextQualityReport"]
