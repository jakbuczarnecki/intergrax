# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Register platform adaptive plugins (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.registry import (
    AdaptiveHealingPluginDescriptor,
    SelfHealingConfidenceEvaluatorRegistry,
    SelfHealingStrategyRankingRegistry,
)
from intergrax.runtime.self_healing.adaptive.confidence_evaluator import AdaptiveConfidenceEvaluator
from intergrax.runtime.self_healing.adaptive.platform_ranking import PlatformHistoricalRankingProvider


def register_platform_adaptive_plugins(
    ranking_registry: SelfHealingStrategyRankingRegistry,
    confidence_registry: SelfHealingConfidenceEvaluatorRegistry,
) -> None:
    ranking = PlatformHistoricalRankingProvider()
    ranking_registry.register(
        ranking,
        AdaptiveHealingPluginDescriptor(
            plugin_id=ranking.provider_id,
            version="1.0.0",
            namespace="platform.self_healing.adaptive",
            priority=100,
            tenant_scope=None,
            timeout_seconds=5.0,
            metadata=(("role", "default_ranking"),),
        ),
    )
    confidence = AdaptiveConfidenceEvaluator()
    confidence_registry.register(
        confidence,
        AdaptiveHealingPluginDescriptor(
            plugin_id=confidence.evaluator_id,
            version="1.0.0",
            namespace="platform.self_healing.adaptive",
            priority=100,
            tenant_scope=None,
            timeout_seconds=5.0,
            metadata=(("role", "default_confidence"),),
        ),
    )


__all__ = ["register_platform_adaptive_plugins"]
