# © Artur Czarnecki. All rights reserved.

"""Built-in insight generators (extend via new classes)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    DecisionOptimizationContext,
    DetectedOptimizationPattern,
    OptimizationArea,
    OptimizationInsight,
)


class PatternLinkageInsightGenerator:
    """Maps each detected pattern to a factual optimization insight."""

    generator_id = "pattern_linkage"
    generator_version = "1"

    def generate_insights(
        self,
        patterns: tuple[DetectedOptimizationPattern, ...],
        *,
        context: DecisionOptimizationContext,
    ) -> tuple[OptimizationInsight, ...]:
        del context
        insights: list[OptimizationInsight] = []
        for pattern in patterns:
            narrative = (
                f"Historical analysis indicates a {pattern.optimization_area.value} "
                f"pattern: {pattern.summary}"
            )
            insights.append(
                OptimizationInsight(
                    insight_id=f"{self.generator_id}:{pattern.pattern_id}",
                    generator_id=self.generator_id,
                    generator_version=self.generator_version,
                    optimization_area=pattern.optimization_area,
                    narrative=narrative,
                    pattern_ids=(pattern.pattern_id,),
                    source_decision_ids=pattern.source_decision_ids,
                    data_source_refs=pattern.data_source_refs,
                    confidence=pattern.confidence,
                )
            )
        return tuple(insights)


class GovernanceFrictionInsightGenerator:
    """Additional narrative for governance friction patterns only."""

    generator_id = "governance_friction_narrative"
    generator_version = "1"

    def generate_insights(
        self,
        patterns: tuple[DetectedOptimizationPattern, ...],
        *,
        context: DecisionOptimizationContext,
    ) -> tuple[OptimizationInsight, ...]:
        del context
        insights: list[OptimizationInsight] = []
        for pattern in patterns:
            if pattern.optimization_area is not OptimizationArea.GOVERNANCE_FRICTION:
                continue
            insights.append(
                OptimizationInsight(
                    insight_id=f"{self.generator_id}:{pattern.pattern_id}",
                    generator_id=self.generator_id,
                    generator_version=self.generator_version,
                    optimization_area=pattern.optimization_area,
                    narrative=(
                        "Manual governance interventions appear frequent relative to allows; "
                        "review routing policies or model choice for affected decisions."
                    ),
                    pattern_ids=(pattern.pattern_id,),
                    source_decision_ids=pattern.source_decision_ids,
                    data_source_refs=pattern.data_source_refs,
                    confidence=pattern.confidence,
                )
            )
        return tuple(insights)


def default_insight_generators() -> tuple[
    PatternLinkageInsightGenerator,
    GovernanceFrictionInsightGenerator,
]:
    return (
        PatternLinkageInsightGenerator(),
        GovernanceFrictionInsightGenerator(),
    )


__all__ = [
    "GovernanceFrictionInsightGenerator",
    "PatternLinkageInsightGenerator",
    "default_insight_generators",
]
