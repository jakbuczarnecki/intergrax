# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-5-R2 hard policy invariant envelope tests."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextProviderContext,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextConflictAction,
    ContextConflictDecision,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
    ContextPolicyDecision,
    ContextPolicyPipelineResult,
    ContextPolicyReasonCode,
    ContextPolicyStage,
    ContextProviderProvenance,
    ContextSemanticDedupDecision,
    replace_context_fragment,
)
from intergrax.context.errors import ContextPolicyInvariantViolationError
from intergrax.context.policy.hard_stages import run_hard_policy_pre_stages
from intergrax.context.policy.invariants import (
    build_fragment_invariant_snapshots,
    validate_policy_pipeline_result,
)
from intergrax.context.policy.pipeline import (
    ContextCrossSourcePolicyPipeline,
    ContextPolicyStrategies,
    default_context_policy_strategies,
)
from intergrax.context.policy.semantic_dedup import DefaultContextSemanticDeduper
from intergrax.context.ranker import DefaultContextRanker
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.data_classification import DataClassification
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
)
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from tests.unit.runtime.nexus.context.test_context_engine import _SmallWindowAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "",
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id=tenant_id,
        user_id=user_id,
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
    authority: ContextAuthorityClass = ContextAuthorityClass.UNASSIGNED,
    scope: ContextFragmentScopeRef | None = None,
    provenance: ContextProviderProvenance | None = None,
    sensitivity: DataClassification = DataClassification.CONFIDENTIAL,
    raw: float | None = 0.95,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id=source_id,
        content=content,
        token_estimate=10,
        relevance_score=0.95,
        freshness_score=0.95,
        confidence_score=0.95,
        mandatory=False,
        authority_class=authority,
        scope_ref=scope or ContextFragmentScopeRef(tenant_id="tenant-a"),
        provider_provenance=provenance,
        sensitivity=sensitivity,
        raw_relevance_signal=raw,
    )


def _prepared(fragments: list[ContextFragment]) -> tuple[list[ContextFragment], dict]:
    hard, _ex, _dec = run_hard_policy_pre_stages(fragments)
    snapshots = build_fragment_invariant_snapshots(hard)
    return hard, snapshots


def _validate_attack(
    fragments: list[ContextFragment],
    result: ContextPolicyPipelineResult,
    *,
    pipeline_id: str = "attack.test",
) -> None:
    hard, snapshots = _prepared(fragments)
    del hard
    validate_policy_pipeline_result(snapshots, result, pipeline_id=pipeline_id)


def test_attack_authority_system_context_to_unassigned() -> None:
    base = _fragment(
        fragment_id="a1",
        content="sys",
        authority=ContextAuthorityClass.SYSTEM_CONTEXT,
    )
    mutated = replace_context_fragment(base, authority_class=ContextAuthorityClass.UNASSIGNED)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_authority_canonical_memory_to_unassigned() -> None:
    base = _fragment(
        fragment_id="a1",
        content="mem",
        authority=ContextAuthorityClass.CANONICAL_MEMORY,
    )
    mutated = replace_context_fragment(base, authority_class=ContextAuthorityClass.UNASSIGNED)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_provenance_stripped() -> None:
    prov = ContextProviderProvenance(provider_id="p1", provider_version="1.0.0", origin="plugin")
    base = _fragment(fragment_id="a1", content="x", provenance=prov)
    mutated = replace_context_fragment(base, provider_provenance=None)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_source_mutation() -> None:
    base = _fragment(fragment_id="a1", content="x", source=ContextFragmentSource.TOOL_OUTPUT)
    mutated = replace_context_fragment(base, source=ContextFragmentSource.SYSTEM_INSTRUCTIONS)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_source_id_changed() -> None:
    base = _fragment(fragment_id="a1", content="x", source_id="orig")
    mutated = replace_context_fragment(base, source_id="forged")
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_sensitivity_downgrade() -> None:
    base = _fragment(
        fragment_id="a1",
        content="x",
        sensitivity=DataClassification.CONFIDENTIAL,
    )
    mutated = replace_context_fragment(base, sensitivity=DataClassification.INTERNAL)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_scope_widening_user_removed() -> None:
    base = _fragment(
        fragment_id="a1",
        content="x",
        scope=ContextFragmentScopeRef(tenant_id="tenant-a", user_id="user-1"),
    )
    mutated = replace_context_fragment(
        base,
        scope_ref=ContextFragmentScopeRef(tenant_id="tenant-a"),
    )
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_scope_execution_cleared() -> None:
    base = _fragment(
        fragment_id="a1",
        content="x",
        scope=ContextFragmentScopeRef(
            tenant_id="tenant-a",
            execution_scope_key="run-1:task-1",
        ),
    )
    mutated = replace_context_fragment(
        base,
        scope_ref=ContextFragmentScopeRef(tenant_id="tenant-a", execution_scope_key=""),
    )
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_unknown_fragment_id() -> None:
    base = _fragment(fragment_id="a1", content="x")
    forged = _fragment(fragment_id="new-id", content="x")
    result = ContextPolicyPipelineResult(fragments=(forged,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_content_mutation() -> None:
    base = _fragment(fragment_id="a1", content="bank balance = 100")
    mutated = replace_context_fragment(base, content="bank balance = 1000000")
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_content_hash_forged() -> None:
    base = _fragment(fragment_id="a1", content="same")
    mutated = replace_context_fragment(
        base,
        content="tampered",
        content_hash=base.content_hash,
    )
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_raw_relevance_signal_changed() -> None:
    base = _fragment(fragment_id="a1", content="x", raw=0.25)
    mutated = replace_context_fragment(base, raw_relevance_signal=0.99, relevance_score=0.99)
    result = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        _validate_attack([base], result)


def test_attack_decision_unknown_fragment_reference() -> None:
    base = _fragment(fragment_id="a1", content="x")
    hard, snapshots = _prepared([base])
    result = ContextPolicyPipelineResult(
        fragments=(hard[0],),
        excluded=(),
        decisions=(
            ContextPolicyDecision(
                stage=ContextPolicyStage.NORMALIZE,
                strategy_id="evil",
                input_fragment_ids=("unknown",),
                output_fragment_ids=("a1",),
                reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
            ),
        ),
    )
    with pytest.raises(ContextPolicyInvariantViolationError):
        validate_policy_pipeline_result(snapshots, result, pipeline_id="evil")


def test_attack_semantic_dedup_unknown_suppressed() -> None:
    base = _fragment(fragment_id="a1", content="x")
    hard, snapshots = _prepared([base])
    result = ContextPolicyPipelineResult(
        fragments=(hard[0],),
        excluded=(),
        decisions=(),
        semantic_dedup_decisions=(
            ContextSemanticDedupDecision(
                kept_fragment_id="a1",
                suppressed_fragment_ids=("ghost",),
                reason_code=ContextPolicyReasonCode.SEMANTIC_DUPLICATE,
                strategy_id="evil",
            ),
        ),
    )
    with pytest.raises(ContextPolicyInvariantViolationError):
        validate_policy_pipeline_result(snapshots, result, pipeline_id="evil")


def test_attack_conflict_unknown_fragment() -> None:
    base = _fragment(fragment_id="a1", content="x")
    hard, snapshots = _prepared([base])
    result = ContextPolicyPipelineResult(
        fragments=(hard[0],),
        excluded=(),
        decisions=(),
        conflict_decisions=(
            ContextConflictDecision(
                left_fragment_id="a1",
                right_fragment_id="ghost",
                action=ContextConflictAction.KEEP_BOTH,
                kept_fragment_ids=("a1",),
                reason_code=ContextPolicyReasonCode.CONFLICT_RESOLVED,
                strategy_id="evil",
            ),
        ),
    )
    with pytest.raises(ContextPolicyInvariantViolationError):
        validate_policy_pipeline_result(snapshots, result, pipeline_id="evil")


def test_positive_policy_score_and_rank_mutations() -> None:
    fragments = [_fragment(fragment_id="a1", content="alpha")]
    hard, snapshots = _prepared(fragments)
    result = ContextCrossSourcePolicyPipeline().execute(hard, _request())
    validate_policy_pipeline_result(snapshots, result, pipeline_id="default")
    assert len(result.fragments) == 1


def test_positive_drop_all_fragments() -> None:
    fragments = [_fragment(fragment_id="a1", content="x")]
    hard, snapshots = _prepared(fragments)

    class EmptyPipeline:
        pipeline_id = "empty.test"

        def execute(self, _fragments, _request, *, strategies=None):
            return ContextPolicyPipelineResult(
                fragments=(),
                excluded=(),
                decisions=(),
            )

    result = EmptyPipeline().execute(hard, _request())
    validate_policy_pipeline_result(snapshots, result, pipeline_id=EmptyPipeline.pipeline_id)
    assert result.fragments == ()


def test_positive_semantic_dedup_subset() -> None:
    first = replace_context_fragment(
        _fragment(fragment_id="a", content="Hello"),
        semantic_fingerprint="fp1",
    )
    second = replace_context_fragment(
        _fragment(fragment_id="b", content="World"),
        semantic_fingerprint="fp1",
    )
    hard, snapshots = _prepared([first, second])
    result = ContextCrossSourcePolicyPipeline().execute(hard, _request())
    validate_policy_pipeline_result(snapshots, result, pipeline_id="default")
    assert len(result.fragments) == 1


def test_positive_custom_ranker() -> None:
    class SpyRanker(DefaultContextRanker):
        calls = 0

        def rank_with_exclusions(self, fragments, request):
            SpyRanker.calls += 1
            return super().rank_with_exclusions(fragments, request)

    strategies = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=strategies.score_normalizer,
        semantic_deduper=strategies.semantic_deduper,
        conflict_resolver=strategies.conflict_resolver,
        ranker=SpyRanker(),
        budget_allocator=strategies.budget_allocator,
    )
    hard, snapshots = _prepared([_fragment(fragment_id="a1", content="x")])
    result = ContextCrossSourcePolicyPipeline(strategies=strategies).execute(hard, _request())
    validate_policy_pipeline_result(snapshots, result, pipeline_id="default")
    assert SpyRanker.calls == 1


def test_positive_custom_semantic_deduper() -> None:
    class SpyDeduper(DefaultContextSemanticDeduper):
        calls = 0

        def deduplicate(self, fragments, request):
            SpyDeduper.calls += 1
            return super().deduplicate(fragments, request)

    strategies = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=strategies.score_normalizer,
        semantic_deduper=SpyDeduper(),
        conflict_resolver=strategies.conflict_resolver,
        ranker=strategies.ranker,
        budget_allocator=strategies.budget_allocator,
    )
    hard, snapshots = _prepared([_fragment(fragment_id="a1", content="x")])
    result = ContextCrossSourcePolicyPipeline(strategies=strategies).execute(hard, _request())
    validate_policy_pipeline_result(snapshots, result, pipeline_id="default")
    assert SpyDeduper.calls == 1


@pytest.mark.asyncio
async def test_engine_sentinel_empty_pipeline_still_legal() -> None:
    class EmptyPipeline:
        pipeline_id = "empty.test.v1"

        def execute(self, fragments, request, *, strategies=None, fragment_budget_tokens=None):
            return ContextPolicyPipelineResult(
                fragments=(),
                excluded=(),
                decisions=(),
            )

    engine = DefaultNexusContextEngine(policy_pipeline=EmptyPipeline())
    request = _request()
    config = RuntimeConfig(llm_adapter=_SmallWindowAdapter(window=512), production_mode=False)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[],
        max_output_tokens=64,
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        runtime=runtime,
        handles={"runtime_config": config, "messages": [], "max_output_tokens": 64},
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.fragments_included == ()
