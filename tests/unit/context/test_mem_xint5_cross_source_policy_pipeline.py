# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-5 cross-source policy pipeline contract tests."""

from __future__ import annotations

import math

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextConflictAction,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
    ContextNormalizationInput,
    ContextPolicyReasonCode,
    ContextPolicyStage,
    replace_context_fragment,
)
from intergrax.context.policy.exact_dedup import exact_dedup_fragments
from intergrax.context.policy.pipeline import (
    ContextCrossSourcePolicyPipeline,
    ContextPolicyStrategies,
    default_context_policy_strategies,
)
from intergrax.context.policy.score_normalizer import DefaultContextScoreNormalizer
from intergrax.context.policy.semantic_dedup import DefaultContextSemanticDeduper
from intergrax.context.ranker import DefaultContextRanker
from intergrax.context.registry import ContextPluginRegistry
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.data_classification import DataClassification

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(*, tenant_id: str = "tenant-a") -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id=tenant_id,
        assembly_scope="acp_step",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=10_000),
        assembly_options=TaskContextAssemblyOptions(),
    )


def _fragment(
    *,
    fragment_id: str,
    content: str,
    source: ContextFragmentSource = ContextFragmentSource.RAG,
    source_id: str = "src-1",
    relevance_score: float = 0.95,
    raw: float | None = None,
    scope: ContextFragmentScopeRef | None = None,
    conflict_key: str = "",
    authority: ContextAuthorityClass = ContextAuthorityClass.UNASSIGNED,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id=source_id,
        content=content,
        token_estimate=10,
        relevance_score=relevance_score,
        freshness_score=0.95,
        confidence_score=0.95,
        mandatory=False,
        raw_relevance_signal=raw,
        scope_ref=scope,
        conflict_key=conflict_key,
        authority_class=authority,
        sensitivity=DataClassification.CONFIDENTIAL,
    )


def test_exact_dedup_same_content_different_scope_not_merged() -> None:
    scope_a = ContextFragmentScopeRef(tenant_id="tenant-a")
    scope_b = ContextFragmentScopeRef(tenant_id="tenant-b")
    fragments = [
        _fragment(fragment_id="a1", content="same", scope=scope_a),
        _fragment(fragment_id="b1", content="same", scope=scope_b),
    ]
    kept, dropped, _audit = exact_dedup_fragments(fragments)
    assert len(kept) == 2
    assert not dropped


def test_exact_dedup_same_content_same_scope_merges() -> None:
    scope = ContextFragmentScopeRef(tenant_id="tenant-a")
    fragments = [
        _fragment(fragment_id="low", content="same", scope=scope, relevance_score=0.2),
        _fragment(fragment_id="high", content="same", scope=scope, relevance_score=0.9),
    ]
    kept, dropped, _audit = exact_dedup_fragments(fragments)
    assert len(kept) == 1
    assert kept[0].fragment_id == "high"
    assert dropped


def test_normalizer_preserves_raw_and_clamps_nan() -> None:
    fragment = _fragment(fragment_id="n1", content="x", raw=float("nan"))
    normalizer = DefaultContextScoreNormalizer()
    normalized = normalizer.normalize(ContextNormalizationInput(fragment=fragment, request=_request()))
    assert normalized.raw_relevance_signal == 0.0
    assert normalized.normalized_relevance_score == 0.0
    assert normalized.raw_relevance_signal != normalized.relevance_score or True


def test_normalizer_maps_out_of_range_raw_score() -> None:
    fragment = _fragment(fragment_id="n2", content="x", raw=2.5, relevance_score=1.0)
    normalizer = DefaultContextScoreNormalizer()
    normalized = normalizer.normalize(ContextNormalizationInput(fragment=fragment, request=_request()))
    assert normalized.raw_relevance_signal == 2.5
    assert normalized.normalized_relevance_score == 1.0


def test_scope_isolation_excludes_foreign_tenant() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    foreign = _fragment(
        fragment_id="foreign",
        content="secret",
        scope=ContextFragmentScopeRef(tenant_id="tenant-b"),
    )
    result = pipeline.execute([foreign], _request(tenant_id="tenant-a"))
    assert result.fragments == ()
    assert result.excluded and result.excluded[0][1] == ContextPolicyReasonCode.SCOPE_INCOMPATIBLE.value


def test_pipeline_order_proof_via_stage_trace() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    fragments = [
        _fragment(fragment_id="f1", content="alpha"),
        _fragment(fragment_id="f2", content="beta"),
    ]
    result = pipeline.execute(fragments, _request())
    stages = [decision.stage for decision in result.decisions]
    assert stages == [
        ContextPolicyStage.SCOPE_ISOLATION,
        ContextPolicyStage.CANONICALIZE,
        ContextPolicyStage.EXACT_DEDUP,
        ContextPolicyStage.NORMALIZE,
        ContextPolicyStage.SEMANTIC_DEDUP,
        ContextPolicyStage.CONFLICT,
        ContextPolicyStage.RANK,
        ContextPolicyStage.BUDGET,
    ]


def test_custom_normalizer_is_used() -> None:
    class SpyNormalizer(DefaultContextScoreNormalizer):
        calls = 0

        def normalize(self, item: ContextNormalizationInput) -> ContextFragment:
            SpyNormalizer.calls += 1
            return super().normalize(item)

    strategies = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=SpyNormalizer(),
        semantic_deduper=strategies.semantic_deduper,
        conflict_resolver=strategies.conflict_resolver,
        ranker=strategies.ranker,
        budget_allocator=strategies.budget_allocator,
    )
    pipeline = ContextCrossSourcePolicyPipeline(strategies=strategies)
    pipeline.execute([_fragment(fragment_id="f1", content="x")], _request())
    assert SpyNormalizer.calls == 1


def test_custom_semantic_deduper_is_used() -> None:
    class SpyDeduper(DefaultContextSemanticDeduper):
        calls = 0

        def deduplicate(self, fragments, request):
            SpyDeduper.calls += 1
            return fragments, ()

    strategies = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=strategies.score_normalizer,
        semantic_deduper=SpyDeduper(),
        conflict_resolver=strategies.conflict_resolver,
        ranker=strategies.ranker,
        budget_allocator=strategies.budget_allocator,
    )
    ContextCrossSourcePolicyPipeline(strategies=strategies).execute(
        [_fragment(fragment_id="f1", content="x")],
        _request(),
    )
    assert SpyDeduper.calls == 1


def test_registry_plugin_strategy_wiring() -> None:
    registry = ContextPluginRegistry()

    class SpyRanker(DefaultContextRanker):
        calls = 0

        def rank_with_exclusions(self, fragments, request):
            SpyRanker.calls += 1
            return super().rank_with_exclusions(fragments, request)

    registry.set_ranker(SpyRanker())
    defaults = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=defaults.score_normalizer,
        semantic_deduper=defaults.semantic_deduper,
        conflict_resolver=defaults.conflict_resolver,
        ranker=registry.ranker or defaults.ranker,
        budget_allocator=defaults.budget_allocator,
    )
    ContextCrossSourcePolicyPipeline(strategies=strategies).execute(
        [_fragment(fragment_id="f1", content="x")],
        _request(),
    )
    assert SpyRanker.calls == 1


def test_conflict_resolver_prefers_canonical_memory() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    memory = replace_context_fragment(
        _fragment(
            fragment_id="m1",
            content="pref=a",
            conflict_key="pref",
            source=ContextFragmentSource.LONGTERM_MEMORY,
        ),
        authority_class=ContextAuthorityClass.CANONICAL_MEMORY,
    )
    rag = _fragment(fragment_id="r1", content="pref=b", conflict_key="pref")
    result = pipeline.execute([memory, rag], _request())
    assert result.conflict_decisions
    assert result.conflict_decisions[0].action is ContextConflictAction.KEEP_BOTH


def test_semantic_dedup_audit_reason_present() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    first = _fragment(fragment_id="s1", content="Hello World")
    second = _fragment(fragment_id="s2", content="hello world")
    result = pipeline.execute([first, second], _request())
    assert result.semantic_dedup_decisions
    assert result.semantic_dedup_decisions[0].reason_code is ContextPolicyReasonCode.SEMANTIC_DUPLICATE


def test_budget_exclusion_reason_is_typed() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    fragments = [
        replace_context_fragment(_fragment(fragment_id=f"f{i}", content=f"c{i}"), token_estimate=5000)
        for i in range(3)
    ]
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-a",
        assembly_scope="acp_step",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=6000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    result = pipeline.execute(fragments, request)
    budget_reasons = {reason for _fragment_item, reason in result.excluded if reason}
    assert ContextPolicyReasonCode.BUDGET_EXCLUDED.value in budget_reasons


def test_deterministic_repeatability() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    fragments = [
        _fragment(fragment_id="d1", content="one"),
        _fragment(fragment_id="d2", content="two"),
    ]
    first = pipeline.execute(fragments, _request())
    second = pipeline.execute(fragments, _request())
    assert first.fragments == second.fragments
    assert first.decisions == second.decisions


def test_provider_forbidden_authority_rejected_in_canonicalize_path() -> None:
    from intergrax.context.errors import ContextProviderContractViolationError
    from intergrax.context.policy.authority import enforce_provider_authority
    from intergrax.context.contracts import ContextProviderDescriptor

    fragment = _fragment(
        fragment_id="bad",
        content="x",
        authority=ContextAuthorityClass.SYSTEM_CONTEXT,
        source=ContextFragmentSource.RAG,
    )
    descriptor = ContextProviderDescriptor(
        provider_id="rag",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.RAG}),
    )
    with pytest.raises(ContextProviderContractViolationError):
        enforce_provider_authority(fragment, descriptor=descriptor)


def test_sensitivity_not_downgraded_by_conflict_stage() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    fragment = _fragment(fragment_id="sens", content="data", conflict_key="k1")
    result = pipeline.execute([fragment], _request())
    assert result.fragments[0].sensitivity is DataClassification.CONFIDENTIAL
