# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy risk evaluation plugin port (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


class AutonomyRiskBand(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class AutonomyRiskFactor:
    label: str
    detail: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label required")
        if not self.detail.strip():
            raise ValueError("detail required")


@dataclass(frozen=True, slots=True)
class AutonomyRiskAssessment:
    evaluator_id: str
    risk_band: AutonomyRiskBand
    factors: tuple[AutonomyRiskFactor, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class AutonomyRiskEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate(
        self,
        request: AutonomyControlRequest,
        effective_level: AutonomyLevel,
    ) -> AutonomyRiskAssessment:
        """Assess risk of acting on the recommendation — read-only, no scoring mandate in R6.1."""
        ...


__all__ = [
    "AutonomyRiskAssessment",
    "AutonomyRiskBand",
    "AutonomyRiskEvaluator",
    "AutonomyRiskFactor",
]
