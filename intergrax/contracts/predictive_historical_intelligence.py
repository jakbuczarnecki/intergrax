# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Historical prediction quality inputs for statistical forecasting (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnalyzerHistoricalReliability:
    """Terminal-outcome precision for one analyzer — not diagnostic truth."""

    analyzer_id: str
    precision: float
    evaluated_count: int

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not (0.0 <= self.precision <= 1.0):
            raise ValueError("precision must be in [0.0, 1.0]")
        if self.evaluated_count < 0:
            raise ValueError("evaluated_count must be non-negative")


@dataclass(frozen=True, slots=True)
class HistoricalRiskIntelligence:
    """
    R2-derived analyzer effectiveness — consumed by R3 confidence calibration only.
    """

    analyzer_reliability: tuple[AnalyzerHistoricalReliability, ...] = ()

    def precision_for(self, analyzer_id: str, *, default: float = 0.5) -> float:
        for entry in self.analyzer_reliability:
            if entry.analyzer_id == analyzer_id:
                return entry.precision
        return default


__all__ = [
    "AnalyzerHistoricalReliability",
    "HistoricalRiskIntelligence",
]
