# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.5 contextual knowledge optimization."""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime, timezone

import pytest

from intergrax.contracts.self_healing.knowledge_evolution import (
    KnowledgeFreshnessPolicy,
    StrategyContextProvider,
    StrategyContextResolutionRequest,
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionTrigger,
    StrategyKnowledgeOperatingContext,
    StrategyKnowledgeContext,
    merge_operating_contexts,
)
from intergrax.contracts.self_healing.knowledge_evolution.comparison import (
    StrategyComparisonScope,
    StrategyComparisonSubject,
)
from intergrax.contracts.self_healing.knowledge_evolution.metrics import (
    StrategyMetricBundle,
    StrategyMetricScope,
    StrategyMetricValue,
)
from intergrax.contracts.self_healing.performance_memory import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.runtime.self_healing.knowledge_evolution import (
    BasicStrategyLearningEngine,
    EvolutionRefsContextProvider,
    InMemoryStrategyKnowledgeRepository,
    NoDecayKnowledgeFreshnessPolicy,
    StrategyKnowledgeEvolutionService,
    SuccessOverSpeedComparisonPolicy,
    SuccessRateMetricProvider,
    TimeWeightedKnowledgeFreshnessPolicy,
)
from intergrax.runtime.self_healing.performance_memory import InMemoryStrategyPerformanceMemoryRepository

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-a"
_STRATEGY = "platform.database.reconnect"
_CONTEXT_FP = "diag:db_timeout:v1"
_CONTEXT_REFS = ("investigation://inv-1",)


def _experience(
    *,
    suffix: str = "1",
    recorded_at: datetime | None = None,
) -> SelfHealingStrategyPerformanceExperience:
    return SelfHealingStrategyPerformanceExperience(
        experience_id=mint_self_healing_strategy_performance_experience_id(),
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        workflow_id="sh_wf_ke000000001",
        plan_id="sh_plan_ke0000001",
        execution_ids=("exec-1",),
        diagnostic_investigation_id="inv-1",
        execution_outcome=SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED,
        rollback_executed=False,
        recovery_time_seconds=2.5,
        evidence_refs=(f"evidence://run/{suffix}",),
        recorded_at=recorded_at or datetime(2026, 6, 1, tzinfo=timezone.utc),
    )


def _scope_context() -> StrategyKnowledgeContext:
    return StrategyKnowledgeContext(
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        context_fingerprint=_CONTEXT_FP,
        context_refs=_CONTEXT_REFS,
        diagnostic_investigation_id="inv-1",
    )


def test_operating_context_creation_and_merge() -> None:
    left = StrategyKnowledgeOperatingContext(
        problem_type="database_timeout",
        environment_label="eu-west",
        provider_id="diag",
    )
    right = StrategyKnowledgeOperatingContext(
        execution_conditions=("high_load",),
        constraints=("read_only",),
        provider_id="infra",
    )
    merged = merge_operating_contexts((left, right))
    assert merged is not None
    assert merged.problem_type == "database_timeout"
    assert merged.environment_label == "eu-west"
    assert merged.execution_conditions == ("high_load",)
    assert merged.constraints == ("read_only",)


def test_operating_context_empty_merge_returns_none() -> None:
    assert merge_operating_contexts(()) is None
    assert merge_operating_contexts((StrategyKnowledgeOperatingContext(),)) is None


def test_evolution_refs_context_provider() -> None:
    provider = EvolutionRefsContextProvider()
    assert isinstance(provider, StrategyContextProvider)
    ctx = provider.resolve(
        StrategyContextResolutionRequest(
            tenant_id=_TENANT,
            strategy_id=_STRATEGY,
            evolution_scope=_scope_context(),
            trigger_refs=("operator://1",),
        ),
    )
    assert ctx is not None
    assert ctx.problem_source == "investigation:inv-1"


def test_freshness_policies_are_interchangeable() -> None:
    no_decay = NoDecayKnowledgeFreshnessPolicy()
    weighted = TimeWeightedKnowledgeFreshnessPolicy(half_life_seconds=3600.0)
    assert isinstance(no_decay, KnowledgeFreshnessPolicy)
    assert isinstance(weighted, KnowledgeFreshnessPolicy)
    ref = datetime(2026, 6, 2, tzinfo=timezone.utc)
    old = datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert no_decay.freshness_score(old, ref) == 1.0
    assert weighted.freshness_score(old, ref) < weighted.freshness_score(ref, ref)


def test_evolve_attaches_operating_context_to_profile() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    service = StrategyKnowledgeEvolutionService(
        performance_memory=memory,
        knowledge_repository=InMemoryStrategyKnowledgeRepository(),
        learning_engine=BasicStrategyLearningEngine(),
        metric_provider=SuccessRateMetricProvider(),
        context_providers=(EvolutionRefsContextProvider(),),
        freshness_policy=NoDecayKnowledgeFreshnessPolicy(),
    )
    result = service.evolve(
        StrategyKnowledgeEvolutionContext(
            knowledge_context=_scope_context(),
            trigger=StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST,
            trigger_refs=("operator://run-1",),
        ),
    )
    assert result.proposed_profile is not None
    assert result.proposed_profile.operating_context is not None
    assert result.proposed_profile.freshness.staleness_policy_id == "platform.no_decay_freshness"


def test_evolve_without_context_providers_leaves_operating_context_none() -> None:
    memory = InMemoryStrategyPerformanceMemoryRepository()
    memory.append(_experience())
    service = StrategyKnowledgeEvolutionService(
        performance_memory=memory,
        knowledge_repository=InMemoryStrategyKnowledgeRepository(),
        learning_engine=BasicStrategyLearningEngine(),
        metric_provider=SuccessRateMetricProvider(),
    )
    result = service.evolve(
        StrategyKnowledgeEvolutionContext(
            knowledge_context=_scope_context(),
            trigger=StrategyKnowledgeEvolutionTrigger.OPERATOR_REQUEST,
            trigger_refs=("operator://run-ctx-none",),
        ),
    )
    assert result.proposed_profile is not None
    assert result.proposed_profile.operating_context is None


def test_comparison_tie_breaks_on_freshness_without_changing_success_winner() -> None:
    policy = SuccessOverSpeedComparisonPolicy()
    metric = StrategyMetricValue(name="success_rate", value=0.5, unit="ratio", evidence_refs=("e1",))
    metric_scope = StrategyMetricScope(
        tenant_id=_TENANT,
        strategy_id=_STRATEGY,
        context_fingerprint=_CONTEXT_FP,
    )
    bundle = StrategyMetricBundle(provider_id="test", scope=metric_scope, metrics=(metric,))
    scope = StrategyComparisonScope(
        tenant_id=_TENANT,
        context_fingerprint=_CONTEXT_FP,
        dimension_weights_ref=None,
    )
    left = StrategyComparisonSubject(
        strategy_id="left",
        metric_bundle=bundle,
        quality_assessment=None,
        knowledge_freshness_score=0.9,
    )
    right = StrategyComparisonSubject(
        strategy_id="right",
        metric_bundle=bundle,
        quality_assessment=None,
        knowledge_freshness_score=0.2,
    )
    result = policy.compare(scope, left, right)
    from intergrax.contracts.self_healing.knowledge_evolution.comparison import StrategyComparisonPreference

    assert result.preference == StrategyComparisonPreference.PREFER_LEFT

    left_win = StrategyComparisonSubject(
        strategy_id="left",
        metric_bundle=StrategyMetricBundle(
            provider_id="test",
            scope=metric_scope,
            metrics=(StrategyMetricValue(name="success_rate", value=0.9, unit="ratio", evidence_refs=("e1",)),),
        ),
        quality_assessment=None,
        knowledge_freshness_score=0.1,
    )
    right_lose = StrategyComparisonSubject(
        strategy_id="right",
        metric_bundle=bundle,
        quality_assessment=None,
        knowledge_freshness_score=0.99,
    )
    winner = policy.compare(scope, left_win, right_lose)
    assert winner.preference == StrategyComparisonPreference.PREFER_LEFT


def test_contextual_contracts_have_no_runtime_imports() -> None:
    modules = (
        "intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context",
        "intergrax.contracts.self_healing.knowledge_evolution.contextual.context_provider",
        "intergrax.contracts.self_healing.knowledge_evolution.contextual.freshness_policy",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        source_path = inspect.getfile(module)
        assert "intergrax\\runtime" not in source_path
        assert "intergrax/runtime" not in source_path
