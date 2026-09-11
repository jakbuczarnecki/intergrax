# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive outcome resolver SPI (PREDICTIVE R5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


@dataclass(frozen=True, slots=True)
class PredictiveOutcomeResolverContext:
    """Readonly facts for outcome evaluation — not diagnostic authority."""

    tenant_id: str
    observed_at: datetime
    evidence_refs: tuple[str, ...]
    execution_failed: bool = False
    problem_created_for_subject: bool = False
    matching_risk_keywords: tuple[str, ...] = ()
    incident_evidence_refs: tuple[str, ...] = ()


class PredictiveOutcomeResolver(Protocol):
    """
    Domain plugin: map predictions + evidence to outcome evaluations.

    Must not create Problems, mutate diagnostic stores, or persist privately.
    """

    @property
    def resolver_id(self) -> str: ...

    def evaluate(
        self,
        prediction: PredictiveRiskSignal,
        context: PredictiveOutcomeResolverContext,
    ) -> PredictionOutcomeEvaluation: ...


__all__ = [
    "PredictiveOutcomeResolver",
    "PredictiveOutcomeResolverContext",
]
