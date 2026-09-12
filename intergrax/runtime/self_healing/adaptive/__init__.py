# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive self-healing intelligence runtime (SELF-HEALING R4)."""

from intergrax.runtime.self_healing.adaptive.bootstrap import register_platform_adaptive_plugins
from intergrax.runtime.self_healing.adaptive.confidence_evaluator import AdaptiveConfidenceEvaluator
from intergrax.runtime.self_healing.adaptive.engine import AdaptiveSelfHealingEngine
from intergrax.runtime.self_healing.adaptive.learning_store import InMemoryAdaptiveHealingLearningRepository
from intergrax.runtime.self_healing.adaptive.platform_ranking import PlatformHistoricalRankingProvider
from intergrax.runtime.self_healing.adaptive.projection import project_adaptive_healing_insights
from intergrax.runtime.self_healing.adaptive.registries import (
    InMemorySelfHealingConfidenceEvaluatorRegistry,
    InMemorySelfHealingStrategyRankingRegistry,
)
from intergrax.runtime.self_healing.adaptive.selector import AdaptiveStrategySelector

__all__ = [
    "AdaptiveConfidenceEvaluator",
    "AdaptiveSelfHealingEngine",
    "AdaptiveStrategySelector",
    "InMemoryAdaptiveHealingLearningRepository",
    "InMemorySelfHealingConfidenceEvaluatorRegistry",
    "InMemorySelfHealingStrategyRankingRegistry",
    "PlatformHistoricalRankingProvider",
    "project_adaptive_healing_insights",
    "register_platform_adaptive_plugins",
]
