# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive healing recommendation — advisory only (SELF-HEALING R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.self_healing.adaptive.score import AdaptiveStrategyScore


class AdaptiveRecommendationStatus(str, Enum):
    OK = "OK"
    DEGRADED_ADAPTIVE_INTELLIGENCE = "DEGRADED_ADAPTIVE_INTELLIGENCE"
    PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class AdaptiveHealingRecommendation:
    """
    Strategy ranking advisory — never an execution or lifecycle decision.
    """

    tenant_id: str
    recommended_strategy_order: tuple[str, ...]
    strategy_scores: tuple[AdaptiveStrategyScore, ...]
    overall_confidence: float
    status: AdaptiveRecommendationStatus
    evidence_refs: tuple[str, ...]
    adaptive_insights: tuple[str, ...] = ()
    confidence_explanation: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if self.status == AdaptiveRecommendationStatus.OK and not self.evidence_refs:
            raise ValueError("evidence_refs required when status is OK")
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError("overall_confidence must be in [0.0, 1.0]")


__all__ = ["AdaptiveHealingRecommendation", "AdaptiveRecommendationStatus"]
