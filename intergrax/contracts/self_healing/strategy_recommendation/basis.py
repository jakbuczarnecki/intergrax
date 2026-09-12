# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation rationale codes (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StrategyRecommendationBasisKind(StrEnum):
    HIGHEST_HISTORICAL_QUALITY_SCORE = "highest_historical_quality_score"
    INSUFFICIENT_HISTORICAL_EVIDENCE = "insufficient_historical_evidence"
    QUALITY_SCORE_TIE_BREAKER = "quality_score_tie_breaker"


@dataclass(frozen=True, slots=True)
class StrategyRecommendationBasis:
    kind: StrategyRecommendationBasisKind
    summary: str

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary required")


__all__ = [
    "StrategyRecommendationBasis",
    "StrategyRecommendationBasisKind",
]
