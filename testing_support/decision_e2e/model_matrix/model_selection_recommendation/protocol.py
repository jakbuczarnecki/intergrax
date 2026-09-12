# © Artur Czarnecki. All rights reserved.

"""Selection strategy plugin contract (DS-E2E-15J-L4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    MatchedCapabilityEvidence,
    ModelSelectionRequest,
    UnmetRequirementEvidence,
)


@dataclass(frozen=True, slots=True)
class StrategyModelAssessment:
    profile_key: str
    eligible: bool
    preference_rank: int
    matched_capabilities: tuple[MatchedCapabilityEvidence, ...]
    unmet_requirements: tuple[UnmetRequirementEvidence, ...]
    outcome_summary: str


@dataclass(frozen=True, slots=True)
class SelectionStrategyResult:
    strategy_id: str
    assessments: tuple[StrategyModelAssessment, ...]


class SelectionStrategy(Protocol):
    """Pluggable model scoring/filtering for one selection concern."""

    @property
    def strategy_id(self) -> str: ...

    def evaluate(
        self,
        request: ModelSelectionRequest,
    ) -> SelectionStrategyResult: ...


__all__ = [
    "SelectionStrategy",
    "SelectionStrategyResult",
    "StrategyModelAssessment",
]
