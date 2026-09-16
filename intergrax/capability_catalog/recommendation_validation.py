# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed recommendation output validation (ME-5)."""

from __future__ import annotations

from intergrax.capability_catalog.errors import CapabilityRecommendationError
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.capability_catalog.recommended_capability import CapabilityRecommendation
from intergrax.contracts.capability_catalog._validation import require_non_empty_text


def validate_recommendation_output(
    *,
    input_ranked: tuple[RankedCapabilityCandidate, ...],
    recommendations: tuple[CapabilityRecommendation, ...],
    recommendation_strategy_id: str,
) -> None:
    """Reject recommendation output that mutates, duplicates, or re-identifies candidates."""
    require_non_empty_text(recommendation_strategy_id, label="recommendation_strategy_id")
    input_by_key = {item.candidate.identity.sort_key: item for item in input_ranked}
    seen_keys: set[tuple[str, str, str, str]] = set()

    for recommendation in recommendations:
        if (
            recommendation.evidence.recommendation_strategy_id
            != recommendation_strategy_id
        ):
            raise CapabilityRecommendationError(
                "recommendation evidence.recommendation_strategy_id must match "
                "the active strategy",
            )
        key = recommendation.ranked.candidate.identity.sort_key
        if key not in input_by_key:
            raise CapabilityRecommendationError(
                "recommendation output contains unknown ranked candidate identity",
            )
        if key in seen_keys:
            raise CapabilityRecommendationError(
                "recommendation output contains duplicate candidate identity",
            )
        seen_keys.add(key)

        original = input_by_key[key]
        if recommendation.ranked != original:
            raise CapabilityRecommendationError(
                "recommendation output must not mutate ranked candidate facts",
            )
