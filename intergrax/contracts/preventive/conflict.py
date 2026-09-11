# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conflicting recommendations are surfaced, never merged (PREVENTIVE R6-Q)."""

from __future__ import annotations

from dataclasses import dataclass


CONFLICTING_RECOMMENDATIONS = "CONFLICTING_RECOMMENDATIONS"


@dataclass(frozen=True, slots=True)
class PreventiveRecommendationConflict:
    """Same scope, incompatible analyzer outputs — operator resolves."""

    marker: str
    tenant_id: str
    risk_signal_id: str
    scope_subject: str
    recommendation_ids: tuple[str, ...]
    analyzer_ids: tuple[str, ...]
    summaries: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.marker != CONFLICTING_RECOMMENDATIONS:
            raise ValueError("marker must be CONFLICTING_RECOMMENDATIONS")
        if len(self.recommendation_ids) < 2:
            raise ValueError("conflict requires at least two recommendations")
        if len(set(self.analyzer_ids)) < 2:
            raise ValueError("conflict requires distinct analyzers")


__all__ = [
    "CONFLICTING_RECOMMENDATIONS",
    "PreventiveRecommendationConflict",
]
