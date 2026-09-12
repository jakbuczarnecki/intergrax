# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive self-healing SPI contracts (SELF-HEALING R4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.adaptive.context import AdaptiveHealingContext
from intergrax.contracts.self_healing.adaptive.score import AdaptiveStrategyScore


@runtime_checkable
class SelfHealingStrategyRankingProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def rank_strategies(
        self,
        context: AdaptiveHealingContext,
    ) -> tuple[AdaptiveStrategyScore, ...]:
        """Return per-strategy scores — no execution surface."""
        ...


@runtime_checkable
class SelfHealingConfidenceEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate_confidence(
        self,
        context: AdaptiveHealingContext,
        strategy_scores: tuple[AdaptiveStrategyScore, ...],
    ) -> tuple[float, tuple[str, ...], str]:
        """
        Return (confidence, evidence_refs, explanation).

        Confidence must be 0.0 when evidence_refs would be empty.
        """
        ...


__all__ = ["SelfHealingConfidenceEvaluator", "SelfHealingStrategyRankingProvider"]
