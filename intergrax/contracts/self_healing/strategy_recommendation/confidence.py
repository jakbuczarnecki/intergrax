# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation confidence labels (SELF-HEALING R5.3)."""

from __future__ import annotations

from enum import StrEnum


class StrategyRecommendationConfidenceLevel(StrEnum):
    """Descriptive certainty — not authorization to execute."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_DATA = "insufficient_data"


__all__ = ["StrategyRecommendationConfidenceLevel"]
