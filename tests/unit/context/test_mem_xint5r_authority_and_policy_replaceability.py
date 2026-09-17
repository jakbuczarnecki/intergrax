# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-5-R authority ownership and policy replaceability closure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
    ContextPolicyPipelineResult,
    ContextPolicyReasonCode,
    ContextProviderContext,
    ContextProviderProvenance,
    replace_context_fragment,
)
from intergrax.context.errors import ContextProviderContractViolationError
from intergrax.context.policy.authority import (
    enforce_provider_authority,
    filter_fragments_by_authority_contract,
)
from intergrax.context.policy.exact_dedup import exact_dedup_fragments
from intergrax.context.policy.pipeline import (
    ContextCrossSourcePolicyPipeline,
    ContextPolicyStrategies,
    default_context_policy_strategies,
)
from intergrax.context.policy.scope_isolation import isolate_assembly_scope
from intergrax.context.policy.semantic_dedup import DefaultContextSemanticDeduper
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.ranker import DefaultContextRanker
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from tests.unit.runtime.nexus.context.test_context_engine import _SmallWindowAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(
    *,
    tenant_id: str = "tenant-a",
    user_id: str = "",
    run_id: str = "run-1",
    task_id: str = "task-1",
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-1",
        run_id=run_id,
        task_id=task_id,
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
    authority: ContextAuthorityClass = ContextAuthorityClass.UNASSIGNED,
    scope: ContextFragmentScopeRef | None = None,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id="src-1",
        content=content,
        token_estimate=10,
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
        authority_class=authority,
        scope_ref=scope,
    )


def test_attack_longterm_source_unassigned_not_promoted_at_collection_boundary() -> None:
    descriptor = build_provider_descriptor(
        "custom.memory",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.LONGTERM_MEMORY}),
        origin="plugin",
    )
    fragment = _fragment(
        fragment_id="attack",
        content="mem",
        source=ContextFragmentSource.LONGTERM_MEMORY,
        authority=ContextAuthorityClass.UNASSIGNED,
    )
    bound = enforce_provider_authority(fragment, descriptor=descriptor)
    assert bound.authority_class is ContextAuthorityClass.UNASSIGNED


def test_pipeline_source_authority_inference_removed() -> None:
    source = Path("intergrax/context/policy/pipeline.py").read_text(encoding="utf-8")
    assert "resolve_authority_for_source" not in source
    assert "AUTHORITY_BY_SOURCE" not in source


def test_external_provider_cannot_self_assign_system_context() -> None:
    descriptor = build_provider_descriptor(
        "custom.attacker",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.SYSTEM_INSTRUCTIONS}),
        origin="plugin",
    )
    fragment = _fragment(
        fragment_id="bad",
        content="x",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        authority=ContextAuthorityClass.SYSTEM_CONTEXT,
    )
    with pytest.raises(ContextProviderContractViolationError):
        enforce_provider_authority(fragment, descriptor=descriptor)


def test_trusted_builtin_system_binding_assigns_system_context() -> None:
    descriptor = build_provider_descriptor(
        "builtin.system_instructions",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.SYSTEM_INSTRUCTIONS}),
        origin="builtin",
    )
    fragment = _fragment(
        fragment_id="sys",
        content="rules",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        authority=ContextAuthorityClass.UNASSIGNED,
    )
    bound = enforce_provider_authority(fragment, descriptor=descriptor)
    assert bound.authority_class is ContextAuthorityClass.SYSTEM_CONTEXT


def test_recording_policy_pipeline_execute_called() -> None:
    class RecordingContextPolicyPipeline:
        pipeline_id = "recording.test.v1"
        execute_calls = 0

        def execute(self, fragments, request, *, strategies=None):
            RecordingContextPolicyPipeline.execute_calls += 1
            return ContextPolicyPipelineResult(
                fragments=(),
                excluded=(),
                decisions=(),
                semantic_dedup_decisions=(),
                conflict_decisions=(),
            )

    recording = RecordingContextPolicyPipeline()
    engine = DefaultNexusContextEngine(policy_pipeline=recording)
    engine._policy_pipeline.execute([_fragment(fragment_id="f1", content="x")], _request())
    assert RecordingContextPolicyPipeline.execute_calls == 1


@pytest.mark.asyncio
async def test_engine_respects_sentinel_pipeline_result() -> None:
    class EmptyPipeline:
        pipeline_id = "empty.test.v1"

        def execute(self, fragments, request, *, strategies=None):
            return ContextPolicyPipelineResult(
                fragments=(),
                excluded=(),
                decisions=(),
                semantic_dedup_decisions=(),
                conflict_decisions=(),
            )

    engine = DefaultNexusContextEngine(policy_pipeline=EmptyPipeline())
    request = _request()
    config = RuntimeConfig(llm_adapter=_SmallWindowAdapter(window=512), production_mode=False)
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={"runtime_config": config, "messages": [], "max_output_tokens": 64},
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.fragments_included == ()


def test_engine_assemble_inner_does_not_recreate_concrete_pipeline() -> None:
    source = Path("intergrax/runtime/nexus/context/context_engine.py").read_text(encoding="utf-8")
    inner = source.split("async def _assemble_inner", maxsplit=1)[1]
    assert "ContextCrossSourcePolicyPipeline(" not in inner


def test_custom_ranker_strategy_still_used_via_execute_kwarg() -> None:
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
    ContextCrossSourcePolicyPipeline().execute(
        [_fragment(fragment_id="f1", content="x")],
        _request(),
        strategies=strategies,
    )
    assert SpyRanker.calls == 1


def test_default_semantic_deduper_is_normalized_fingerprint_not_embedding() -> None:
    deduper = DefaultContextSemanticDeduper()
    assert "normalized_fingerprint" in deduper.strategy_id
    first = replace_context_fragment(
        _fragment(fragment_id="a", content="Hello"),
        semantic_fingerprint="fp1",
    )
    second = replace_context_fragment(
        _fragment(fragment_id="b", content="World"),
        semantic_fingerprint="fp1",
    )
    kept, decisions = deduper.deduplicate([first, second], _request())
    assert len(kept) == 1
    assert decisions


def test_semantic_dedup_dead_config_removed() -> None:
    import intergrax.context.policy.semantic_dedup as semantic_dedup_module

    assert not hasattr(semantic_dedup_module, "SemanticDedupPolicyConfig")


def test_scope_isolation_rejects_foreign_tenant() -> None:
    foreign = _fragment(
        fragment_id="foreign",
        content="x",
        scope=ContextFragmentScopeRef(tenant_id="tenant-b"),
    )
    kept, excluded = isolate_assembly_scope([foreign], _request(tenant_id="tenant-a"))
    assert kept == []
    assert excluded[0][1] == ContextPolicyReasonCode.SCOPE_INCOMPATIBLE.value


def test_scope_isolation_rejects_cross_user_when_request_user_set() -> None:
    foreign_user = _fragment(
        fragment_id="u2",
        content="x",
        scope=ContextFragmentScopeRef(
            tenant_id="tenant-a",
            user_id="user-2",
            execution_scope_key="run-1:task-1",
        ),
    )
    kept, excluded = isolate_assembly_scope([foreign_user], _request(user_id="user-1"))
    assert kept == []
    assert excluded


def test_scope_isolation_rejects_cross_execution_scope() -> None:
    fragment = _fragment(
        fragment_id="exec",
        content="x",
        scope=ContextFragmentScopeRef(
            tenant_id="tenant-a",
            execution_scope_key="other-run:other-task",
        ),
    )
    kept, excluded = isolate_assembly_scope([fragment], _request())
    assert kept == []
    assert excluded


def test_exact_dedup_retention_uses_explicit_authority_only() -> None:
    low = replace_context_fragment(
        _fragment(fragment_id="low", content="same canonical", source=ContextFragmentSource.RAG),
        authority_class=ContextAuthorityClass.UNASSIGNED,
        semantic_fingerprint="fp",
    )
    high = replace_context_fragment(
        _fragment(fragment_id="high", content="same canonical", source=ContextFragmentSource.LONGTERM_MEMORY),
        authority_class=ContextAuthorityClass.RAG_EVIDENCE,
        semantic_fingerprint="fp",
    )
    kept, _dropped, _audit = exact_dedup_fragments([low, high])
    assert len(kept) == 1
    assert kept[0].fragment_id == "high"


def test_custom_pipeline_cannot_bypass_post_authority_gate() -> None:
    descriptor = build_provider_descriptor(
        "custom.attacker",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.RAG}),
        origin="plugin",
    )
    elevated = replace_context_fragment(
        _fragment(fragment_id="elev", content="x"),
        authority_class=ContextAuthorityClass.SYSTEM_CONTEXT,
        provider_provenance=ContextProviderProvenance.from_descriptor(descriptor),
    )
    kept, excluded = filter_fragments_by_authority_contract(
        [elevated],
        descriptors_by_id={descriptor.provider_id: descriptor},
    )
    assert kept == []
    assert excluded[0][1] == "authority.contract_violation"
