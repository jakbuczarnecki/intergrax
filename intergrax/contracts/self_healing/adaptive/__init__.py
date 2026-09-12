# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive self-healing intelligence contracts (SELF-HEALING R4)."""

from intergrax.contracts.self_healing.adaptive.context import (
    AdaptiveHealingContext,
    AdaptiveRollbackHistoryEntry,
    AdaptiveValidationQualitySignal,
)
from intergrax.contracts.self_healing.adaptive.errors import (
    DEGRADED_ADAPTIVE_INTELLIGENCE,
    PLUGIN_UNAVAILABLE,
)
from intergrax.contracts.self_healing.adaptive.learning import AdaptiveHealingLearningRepository
from intergrax.contracts.self_healing.adaptive.recommendation import (
    AdaptiveHealingRecommendation,
    AdaptiveRecommendationStatus,
)
from intergrax.contracts.self_healing.adaptive.registry import (
    AdaptiveHealingPluginDescriptor,
    SelfHealingConfidenceEvaluatorRegistry,
    SelfHealingStrategyRankingRegistry,
)
from intergrax.contracts.self_healing.adaptive.score import AdaptiveScoringFactor, AdaptiveStrategyScore
from intergrax.contracts.self_healing.adaptive.spi import (
    SelfHealingConfidenceEvaluator,
    SelfHealingStrategyRankingProvider,
)

__all__ = [
    "AdaptiveHealingContext",
    "AdaptiveHealingLearningRepository",
    "AdaptiveHealingPluginDescriptor",
    "AdaptiveHealingRecommendation",
    "AdaptiveRecommendationStatus",
    "AdaptiveRollbackHistoryEntry",
    "AdaptiveScoringFactor",
    "AdaptiveStrategyScore",
    "AdaptiveValidationQualitySignal",
    "DEGRADED_ADAPTIVE_INTELLIGENCE",
    "PLUGIN_UNAVAILABLE",
    "SelfHealingConfidenceEvaluator",
    "SelfHealingConfidenceEvaluatorRegistry",
    "SelfHealingStrategyRankingProvider",
    "SelfHealingStrategyRankingRegistry",
]
