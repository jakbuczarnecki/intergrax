# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Confidence descriptors for autonomy decision evaluation (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AutonomyEvaluationConfidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class AutonomyEvaluationConfidence:
    level: AutonomyEvaluationConfidenceLevel
    score: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")


__all__ = [
    "AutonomyEvaluationConfidence",
    "AutonomyEvaluationConfidenceLevel",
]
