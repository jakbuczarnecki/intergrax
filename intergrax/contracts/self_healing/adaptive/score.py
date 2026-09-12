# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy scoring models for adaptive healing (SELF-HEALING R4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveScoringFactor:
    factor_id: str
    weight: float
    raw_value: float

    def __post_init__(self) -> None:
        if not self.factor_id.strip():
            raise ValueError("factor_id required")


@dataclass(frozen=True, slots=True)
class AdaptiveStrategyScore:
    strategy_id: str
    calculated_score: float
    confidence: float
    evidence_refs: tuple[str, ...]
    scoring_factors: tuple[AdaptiveScoringFactor, ...]

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.scoring_factors:
            raise ValueError("scoring_factors must be non-empty")


__all__ = ["AdaptiveScoringFactor", "AdaptiveStrategyScore"]
