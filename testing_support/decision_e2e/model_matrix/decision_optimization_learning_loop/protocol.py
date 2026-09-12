# © Artur Czarnecki. All rights reserved.

"""Pluggable decision optimization contracts (DS-E2E-15J-L9)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    DecisionOptimizationContext,
    DecisionOptimizationSuggestion,
    DetectedOptimizationPattern,
    OptimizationInsight,
)


class OptimizationPatternAnalyzer(Protocol):
    """Pluggable pattern detection over optimization context."""

    @property
    def analyzer_id(self) -> str: ...

    @property
    def analyzer_version(self) -> str: ...

    def detect_patterns(
        self,
        context: DecisionOptimizationContext,
    ) -> tuple[DetectedOptimizationPattern, ...]: ...


class OptimizationInsightGenerator(Protocol):
    """Pluggable insight synthesis from detected patterns."""

    @property
    def generator_id(self) -> str: ...

    @property
    def generator_version(self) -> str: ...

    def generate_insights(
        self,
        patterns: tuple[DetectedOptimizationPattern, ...],
        *,
        context: DecisionOptimizationContext,
    ) -> tuple[OptimizationInsight, ...]: ...


class OptimizationRecommendationProvider(Protocol):
    """Pluggable human-facing recommendations from insights."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def recommend(
        self,
        insights: tuple[OptimizationInsight, ...],
        *,
        context: DecisionOptimizationContext,
        generated_at: datetime,
    ) -> tuple[DecisionOptimizationSuggestion, ...]: ...


__all__ = [
    "OptimizationInsightGenerator",
    "OptimizationPatternAnalyzer",
    "OptimizationRecommendationProvider",
]
