# © Artur Czarnecki. All rights reserved.

"""AUTONOMOUS-ENTERPRISE-SELF-HEALING-ADAPTIVE-INTELLIGENCE R4 qualification matrix."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.self_healing.adaptive.context import AdaptiveHealingContext
from intergrax.contracts.self_healing.adaptive.registry import AdaptiveHealingPluginDescriptor
from intergrax.contracts.self_healing.adaptive.recommendation import AdaptiveRecommendationStatus
from intergrax.contracts.self_healing.adaptive.score import AdaptiveScoringFactor, AdaptiveStrategyScore
from intergrax.runtime.self_healing import SelfHealingDecisionEngine
from intergrax.runtime.self_healing.adaptive import (
    AdaptiveConfidenceEvaluator,
    AdaptiveSelfHealingEngine,
    AdaptiveStrategySelector,
    InMemorySelfHealingConfidenceEvaluatorRegistry,
    InMemorySelfHealingStrategyRankingRegistry,
    project_adaptive_healing_insights,
    register_platform_adaptive_plugins,
)
from intergrax.runtime.self_healing.defaults import platform_default_strategies
from intergrax.runtime.self_healing.strategy_registry import InMemorySelfHealingStrategyRegistry
from tests.unit.runtime.self_healing.test_autonomous_enterprise_self_healing_strategy_r1_q import (
    _sample_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _ReverseRankingProvider:
    provider_id = "test.reverse_ranking"

    def rank_strategies(self, context: AdaptiveHealingContext) -> tuple[AdaptiveStrategyScore, ...]:
        scores: list[AdaptiveStrategyScore] = []
        for index, strategy in enumerate(context.strategy_candidates):
            scores.append(
                AdaptiveStrategyScore(
                    strategy_id=strategy.strategy_id,
                    calculated_score=float(index + 1),
                    confidence=0.9,
                    evidence_refs=context.evidence_refs,
                    scoring_factors=(AdaptiveScoringFactor("test_boost", 1.0, float(index + 1)),),
                ),
            )
        return tuple(scores)


class _ExplodingRankingProvider:
    provider_id = "test.exploding_ranking"

    def rank_strategies(self, context: AdaptiveHealingContext) -> tuple[AdaptiveStrategyScore, ...]:
        raise RuntimeError("ranking plugin exploded")


def _adaptive_stack(
    *,
    ranking_registry: InMemorySelfHealingStrategyRankingRegistry | None = None,
) -> tuple[AdaptiveSelfHealingEngine, InMemorySelfHealingStrategyRankingRegistry]:
    ranking = ranking_registry or InMemorySelfHealingStrategyRankingRegistry()
    confidence = InMemorySelfHealingConfidenceEvaluatorRegistry()
    register_platform_adaptive_plugins(ranking, confidence)
    engine = AdaptiveSelfHealingEngine(ranking, confidence)
    return engine, ranking


def test_custom_strategy_ranking_provider() -> None:
    ranking = InMemorySelfHealingStrategyRankingRegistry()
    confidence = InMemorySelfHealingConfidenceEvaluatorRegistry()
    engine = AdaptiveSelfHealingEngine(ranking, confidence)
    custom = _ReverseRankingProvider()
    ranking.register(
        custom,
        AdaptiveHealingPluginDescriptor(
            plugin_id=custom.provider_id,
            version="1.0.0",
            namespace="test.adaptive",
            priority=200,
            tenant_scope=None,
            timeout_seconds=2.0,
        ),
    )
    context = _sample_context()
    strategies = platform_default_strategies()
    adaptive_context = engine.build_context(context, strategies)
    recommendation = engine.recommend(adaptive_context)
    assert recommendation.recommended_strategy_order[0] == strategies[-1].strategy_id
    insights = project_adaptive_healing_insights(recommendation)
    assert insights[0].strategy_ranking_summary


def test_strategy_failure_isolated() -> None:
    engine, ranking = _adaptive_stack()
    bad = _ExplodingRankingProvider()
    ranking.register(
        bad,
        AdaptiveHealingPluginDescriptor(
            plugin_id=bad.provider_id,
            version="1.0.0",
            namespace="test.adaptive",
            priority=300,
            tenant_scope=None,
            timeout_seconds=2.0,
        ),
    )
    adaptive_context = engine.build_context(_sample_context(), platform_default_strategies())
    recommendation = engine.recommend(adaptive_context)
    assert recommendation.status in {
        AdaptiveRecommendationStatus.DEGRADED_ADAPTIVE_INTELLIGENCE,
        AdaptiveRecommendationStatus.OK,
    }
    assert recommendation.recommended_strategy_order


def test_confidence_requires_evidence() -> None:
    evaluator = AdaptiveConfidenceEvaluator()
    empty_context = AdaptiveHealingContext(
        tenant_id="tenant_a",
        strategy_candidates=(),
        historical_outcomes=(),
        evidence_refs=(),
        execution_context_ref=None,
    )
    confidence, evidence, explanation = evaluator.evaluate_confidence(empty_context, ())
    assert confidence == 0.0
    assert evidence == ()
    assert "evidence" in explanation


def test_adaptive_layer_cannot_execute() -> None:
    adaptive_root = Path("intergrax/runtime/self_healing/adaptive")
    forbidden_import_markers = (
        "external_operations.admission.execution_gate",
        "ExternalOperationExecutionGate",
        "SelfHealingLifecycleEngine",
        "execute_contained_provider_call",
    )
    for path in adaptive_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for marker in forbidden_import_markers:
                    assert marker not in (node.module or ""), f"{path} imports execution spine"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for marker in forbidden_import_markers:
                        assert marker not in alias.name, f"{path} imports execution spine"
    engine, ranking = _adaptive_stack()
    selector = AdaptiveStrategySelector(engine)
    registry = InMemorySelfHealingStrategyRegistry()
    for strategy in platform_default_strategies():
        registry.register(strategy)
    decision_engine = SelfHealingDecisionEngine(registry, strategy_selector=selector)
    results = decision_engine.select_and_evaluate(_sample_context())
    assert results


class _TenantOnlyRankingProvider:
    provider_id = "test.tenant_only_ranking"

    def rank_strategies(self, context: AdaptiveHealingContext) -> tuple[AdaptiveStrategyScore, ...]:
        return (
            AdaptiveStrategyScore(
                strategy_id=context.strategy_candidates[0].strategy_id,
                calculated_score=1.0,
                confidence=0.5,
                evidence_refs=context.evidence_refs,
                scoring_factors=(AdaptiveScoringFactor("tenant_only", 1.0, 1.0),),
            ),
        )


def test_tenant_isolation() -> None:
    engine, ranking = _adaptive_stack()
    tenant_only = _TenantOnlyRankingProvider()
    ranking.register(
        tenant_only,
        AdaptiveHealingPluginDescriptor(
            plugin_id=tenant_only.provider_id,
            version="1.0.0",
            namespace="test.adaptive",
            priority=500,
            tenant_scope=frozenset({"tenant_b"}),
            timeout_seconds=2.0,
        ),
    )
    context_a = engine.build_context(_sample_context(tenant_id="tenant_a"), platform_default_strategies())
    providers_a = ranking.list_for_tenant("tenant_a")
    provider_ids_a = {descriptor.plugin_id for _, descriptor in providers_a}
    assert "test.tenant_only_ranking" not in provider_ids_a
    recommendation_a = engine.recommend(context_a)
    assert recommendation_a.tenant_id == "tenant_a"

    with pytest.raises(ValueError, match="tenant isolation"):
        AdaptiveHealingContext(
            tenant_id="tenant_b",
            strategy_candidates=platform_default_strategies(),
            historical_outcomes=_sample_context(tenant_id="tenant_a").historical_outcomes,
            evidence_refs=("evidence://b",),
            execution_context_ref=None,
        )


def test_existing_self_healing_regression() -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_strategy_r1_q.py",
        "tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_orchestration_r2_q.py",
        "tests/unit/runtime/self_healing/test_autonomous_enterprise_self_healing_orchestration_r3_q.py",
        "-q",
        "--tb=no",
    ]
    completed = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
