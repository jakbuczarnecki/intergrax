# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Knowledge confidence labels (SELF-HEALING R5.4)."""

from __future__ import annotations

from enum import StrEnum


class StrategyKnowledgeConfidenceLevel(StrEnum):
    """Epistemic strength — not authorization to execute."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_DATA = "insufficient_data"


__all__ = ["StrategyKnowledgeConfidenceLevel"]
