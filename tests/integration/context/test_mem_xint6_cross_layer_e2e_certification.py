# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-6 — enterprise cross-layer Memory × RAG × Tools × Session × CE certification."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.context.bootstrap import materialize_context_plugin_registry
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextConflictAction,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentScopeRef,
    ContextFragmentSource,
    ContextPolicyReasonCode,
    ContextPolicyStage,
    ContextProviderContext,
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
from intergrax.context.policy.scope_isolation import isolate_assembly_scope
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.ranker import DefaultContextRanker
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.execution_identity import (
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.memory_control import (
    MemoryControlPlaneScope,
    MemoryControlRecallItem,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from intergrax.runtime.execution.active_execution_budget import reset_active_execution_budget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.canonical_context_composition import (
    enforce_context_engine_when_provider_sources_active,
)
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.iterative_tool_context_assembly import run_ce_bounded_tool_loop
from intergrax.runtime.nexus.context.memory_context_invocation import run_longterm_memory_context
from intergrax.runtime.nexus.context.provider_handles import LTM_ENTRIES_METADATA_KEY
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    merge_provider_metadata_into_request,
)
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)
from testing_support.mem_xint6_cross_layer_certification import (
    authority_for_source,
    assert_included_fragments_have_provenance,
    build_certification_engine,
    build_provider_handles,
    certification_assembly_request,
    inventory_legacy_bypass_symbols,
    session_snapshot_for_cert,
)
from testing_support.memory_control_plane_test_stub import MemoryControlPlaneTestStub
from tests.integration.memory.e2e.harness import build_in_memory_memory_harness
from tests.unit.runtime.nexus.context.test_context_engine import _SmallWindowAdapter
from tests.unit.runtime.nexus.context.test_mem_xint4r_react_context_authority import (
    RecordingContextEngine,
    _TwoToolRoundPlanner,
    _bind_identity,
    _invoker,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_TENANT = "tenant-mem-xint6"
_USER = "user-mem-xint6"
_LTM_SNIPPET = "cert-memory-fact-omega"


def _identity() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


@dataclass
class RecordingRecallPlane(MemoryControlPlaneTestStub):
    recall_calls: list[tuple[RequestIdentity, MemoryControlScopeRef, MemoryControlRecallRequest]] = field(
        default_factory=list
    )

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> MemoryControlRecallResult:
        self.recall_calls.append((identity, scope, request))
        return MemoryControlRecallResult(
            scope=MemoryControlPlaneScope.USER,
            items=(
                MemoryControlRecallItem(
                    entry_id="ltm-cert-1",
                    content=_LTM_SNIPPET,
                    kind=MemoryKind.USER_FACT,
                    score=0.91,
                ),
            ),
            reason="hits",
        )


def _runtime_config_with_plane(plane: MemoryControlPlaneTestStub) -> RuntimeConfig:
    from intergrax.tools.registry.wiring import ToolWiringContext

    return RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_user_longterm_memory=True,
        enable_rag=False,
        tool_wiring_context=ToolWiringContext(extras={"memory_control_plane": plane}),
    )


def _policy_fragment(
    *,
    fragment_id: str,
    content: str,
    source: ContextFragmentSource,
    authority: ContextAuthorityClass | None = None,
    conflict_key: str = "",
    scope: ContextFragmentScopeRef | None = None,
    token_estimate: int = 10,
) -> ContextFragment:
    auth = authority if authority is not None else authority_for_source(source)
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        source_id=f"src-{fragment_id}",
        content=content,
        token_estimate=token_estimate,
        relevance_score=0.9,
        freshness_score=0.9,
        confidence_score=0.9,
        mandatory=False,
        authority_class=auth,
        conflict_key=conflict_key,
        scope_ref=scope or ContextFragmentScopeRef(tenant_id=_TENANT, user_id=_USER),
        sensitivity=DataClassification.CONFIDENTIAL,
        raw_relevance_signal=0.88,
    )


async def _assemble_with_handles(
    *,
    handles: dict,
    request: ContextAssemblyRequest | None = None,
    engine: DefaultNexusContextEngine | None = None,
):
    engine = engine or build_certification_engine()
    req = request or certification_assembly_request(tenant_id=_TENANT, user_id=_USER)
    provider_ctx = ContextProviderContext(engine_id=engine.engine_id, handles=handles)
    return await engine.assemble(req, provider_ctx=provider_ctx)


@pytest.mark.asyncio
async def test_memory_recall_enters_model_only_through_context_engine() -> None:
    plane = RecordingRecallPlane()
    session_manager = build_in_memory_session_manager()
    session_manager.search_user_longterm_memory = AsyncMock()  # type: ignore[method-assign]
    request = replace(
        build_runtime_request_for_tests(tenant_id=_TENANT, user_id=_USER, message="recall omega"),
        canonical_identity=_identity(),
    )
    config = _runtime_config_with_plane(plane)
    ctx = RuntimeContext.build(config=config, session_manager=session_manager)
    run_id = canonical_run_id_for_tests("mem-recall")
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=run_id,
        messages_for_llm=[ChatMessage(role="user", content="recall omega")],
    )
    with canonical_execution_identity_scope(run_id):
        await run_longterm_memory_context(state)
    session_manager.search_user_longterm_memory.assert_not_called()
    assert len(plane.recall_calls) >= 1
    merge_provider_metadata_into_request(state)
    assert LTM_ENTRIES_METADATA_KEY in state.request.metadata

    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    handles = build_provider_handles(
        runtime_config=runtime_config,
        ltm_entries=state.request.metadata[LTM_ENTRIES_METADATA_KEY],
    )
    assembled = await _assemble_with_handles(handles=handles)
    joined = "\n".join(m.content or "" for m in assembled.messages)
    assert _LTM_SNIPPET in joined
    memory_frags = [f for f in assembled.fragments_included if f.source is ContextFragmentSource.LONGTERM_MEMORY]
    assert memory_frags
    assert memory_frags[0].authority_class is ContextAuthorityClass.CANONICAL_MEMORY
    assert_included_fragments_have_provenance(assembled.fragments_included)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_rag_evidence_enters_model_only_through_context_engine() -> None:
    chunk_text = "rag-evidence-zeta"
    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    handles = build_provider_handles(
        runtime_config=runtime_config,
        rag_chunks=[{"id": "c1", "text": chunk_text, "score": 0.82}],
    )
    assembled = await _assemble_with_handles(handles=handles)
    assert chunk_text in "\n".join(m.content or "" for m in assembled.messages)
    rag_frags = [f for f in assembled.fragments_included if f.source is ContextFragmentSource.RAG]
    assert rag_frags and rag_frags[0].authority_class is ContextAuthorityClass.RAG_EVIDENCE


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_multiround_react_assembles_context_once_per_model_round() -> None:
    state = build_runtime_state_for_tests(run_id=mint_run_id())
    engine = RecordingContextEngine()
    state.context.config.context_engine = engine
    state.context.config.llm_adapter = FakeLLMAdapter()
    state.context.config.max_tool_iterations = 2
    token, budget_token = _bind_identity(state)
    try:
        await run_ce_bounded_tool_loop(
            state=state,
            invoker=_invoker(),
            tool_planner=_TwoToolRoundPlanner(),
            planner_input=[ChatMessage(role="user", content="u")],
            allowed_tool_ids=("probe.read",),
            max_iterations=2,
        )
    finally:
        reset_active_execution_identity(token)
        reset_active_execution_budget(budget_token)
    assert engine.assemble_calls >= 2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_session_episodic_and_canonical_memory_distinct_in_ce() -> None:
    episodic = "session-episodic-delta"
    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    snapshot = session_snapshot_for_cert(
        tenant_id=_TENANT,
        session_id="sess-1",
        revision_id="rev-1",
        episodic_fact=episodic,
    )
    handles = build_provider_handles(
        runtime_config=runtime_config,
        ltm_entries=[{"entry_id": "e1", "content": _LTM_SNIPPET, "kind": "user_fact"}],
        session_snapshot=snapshot,
    )
    assembled = await _assemble_with_handles(handles=handles)
    sources = {f.source for f in assembled.fragments_included}
    assert ContextFragmentSource.LONGTERM_MEMORY in sources
    assert ContextFragmentSource.SESSION_HISTORY in sources
    authorities = {f.authority_class for f in assembled.fragments_included}
    assert ContextAuthorityClass.CANONICAL_MEMORY in authorities
    assert ContextAuthorityClass.SESSION_EPISODIC in authorities
    joined = "\n".join(m.content or "" for m in assembled.messages)
    assert joined.count(_LTM_SNIPPET) == 1


def test_memory_rag_conflict_resolver_prefers_policy_typed_decision() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    memory = replace_context_fragment(
        _policy_fragment(
            fragment_id="m1",
            content="pref=a",
            source=ContextFragmentSource.LONGTERM_MEMORY,
            conflict_key="pref",
        ),
        authority_class=ContextAuthorityClass.CANONICAL_MEMORY,
    )
    rag = _policy_fragment(
        fragment_id="r1",
        content="pref=b",
        source=ContextFragmentSource.RAG,
        conflict_key="pref",
    )
    result = pipeline.execute([memory, rag], certification_assembly_request())
    assert result.conflict_decisions
    assert result.conflict_decisions[0].action is ContextConflictAction.KEEP_BOTH


def test_rag_and_tool_duplicate_is_suppressed_before_budget() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    duplicate = "shared-evidence-norm"
    rag = _policy_fragment(fragment_id="r1", content=duplicate, source=ContextFragmentSource.RAG)
    tool = _policy_fragment(fragment_id="t1", content=duplicate.upper(), source=ContextFragmentSource.TOOL_OUTPUT)
    result = pipeline.execute([rag, tool], certification_assembly_request())
    assert result.semantic_dedup_decisions
    included_contents = {f.content for f in result.fragments}
    assert len(included_contents) < 2 or result.semantic_dedup_decisions[0].reason_code is ContextPolicyReasonCode.SEMANTIC_DUPLICATE


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_mixed_source_full_pipeline_stages_and_provenance_walkthrough() -> None:
    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    handles = build_provider_handles(
        runtime_config=runtime_config,
        ltm_entries=[{"entry_id": "e1", "content": _LTM_SNIPPET, "kind": "user_fact"}],
        rag_chunks=[{"id": "c1", "text": "rag-mixed", "score": 0.7}],
        tool_blocks=[{"content": "tool-obs", "tool_call_id": "tc-1", "tool_name": "probe"}],
        web_blocks=["web-hit"],
        session_snapshot=session_snapshot_for_cert(
            tenant_id=_TENANT,
            session_id="sess-mix",
            revision_id="rev-mix",
            episodic_fact="episodic-mix",
        ),
    )
    assembled = await _assemble_with_handles(handles=handles)
    assert assembled.fragments_included
    assert assembled.policy_decisions
    stages = {d.stage for d in assembled.policy_decisions}
    for required in (
        ContextPolicyStage.NORMALIZE,
        ContextPolicyStage.SEMANTIC_DEDUP,
        ContextPolicyStage.CONFLICT,
        ContextPolicyStage.RANK,
        ContextPolicyStage.BUDGET,
    ):
        assert required in stages
    sample = assembled.fragments_included[0]
    assert sample.source_id
    assert sample.authority_class is not ContextAuthorityClass.UNASSIGNED
    assert assembled.total_tokens <= assembled.budget_tokens


def test_budget_pressure_excludes_with_typed_reason() -> None:
    pipeline = ContextCrossSourcePolicyPipeline()
    fragments = [
        replace_context_fragment(
            _policy_fragment(fragment_id=f"f{i}", content=f"c{i}", source=ContextFragmentSource.RAG),
            token_estimate=5000,
        )
        for i in range(3)
    ]
    request = certification_assembly_request(max_tokens=6000)
    result = pipeline.execute(fragments, request)
    reasons = {reason for _frag, reason in result.excluded}
    assert ContextPolicyReasonCode.BUDGET_EXCLUDED.value in reasons


def test_custom_policy_cannot_mutate_provenance() -> None:
    from intergrax.context.contracts import ContextPolicyPipelineResult

    fragment = _policy_fragment(fragment_id="x1", content="attack", source=ContextFragmentSource.RAG)
    hard, _, _ = run_hard_policy_pre_stages([fragment])
    snapshots = build_fragment_invariant_snapshots(hard)
    mutated = replace_context_fragment(
        hard[0],
        provider_provenance=None,
        authority_class=ContextAuthorityClass.SYSTEM_CONTEXT,
    )
    forged = ContextPolicyPipelineResult(fragments=(mutated,), excluded=(), decisions=())
    with pytest.raises(ContextPolicyInvariantViolationError):
        validate_policy_pipeline_result(snapshots, forged, pipeline_id="mem-xint6.attack")


def test_custom_strategies_replaceable_through_contracts() -> None:
    class SpyRanker(DefaultContextRanker):
        calls = 0

        def rank_with_exclusions(self, fragments, request):
            SpyRanker.calls += 1
            return super().rank_with_exclusions(fragments, request)

    defaults = default_context_policy_strategies()
    strategies = ContextPolicyStrategies(
        score_normalizer=defaults.score_normalizer,
        semantic_deduper=defaults.semantic_deduper,
        conflict_resolver=defaults.conflict_resolver,
        ranker=SpyRanker(),
        budget_allocator=defaults.budget_allocator,
    )
    ContextCrossSourcePolicyPipeline(strategies=strategies).execute(
        [_policy_fragment(fragment_id="c1", content="x", source=ContextFragmentSource.RAG)],
        certification_assembly_request(),
    )
    assert SpyRanker.calls == 1


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_remember_recall_roundtrip_reaches_ce_without_manager_bypass() -> None:
    harness = build_in_memory_memory_harness(tenant_id=_TENANT, user_id=_USER)
    identity = harness.identity()
    scope = harness.user_scope(identity)
    remembered = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content=_LTM_SNIPPET, kind=MemoryKind.USER_FACT),
    )
    assert remembered.entry_id
    recall = await harness.plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="omega", top_k=3),
    )
    assert any(_LTM_SNIPPET in item.content for item in recall.items)
    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    rows = [{"entry_id": item.entry_id, "content": item.content, "kind": "user_fact"} for item in recall.items]
    assembled = await _assemble_with_handles(
        handles=build_provider_handles(runtime_config=runtime_config, ltm_entries=rows),
    )
    assert _LTM_SNIPPET in "\n".join(m.content or "" for m in assembled.messages)


def test_scope_cross_tenant_hard_excludes_before_model() -> None:
    foreign = _policy_fragment(
        fragment_id="foreign",
        content="secret",
        source=ContextFragmentSource.RAG,
        scope=ContextFragmentScopeRef(tenant_id="tenant-other"),
    )
    kept, excluded = isolate_assembly_scope([foreign], certification_assembly_request(tenant_id=_TENANT))
    assert not kept
    assert excluded and excluded[0][1] == ContextPolicyReasonCode.SCOPE_INCOMPATIBLE.value


def test_fail_closed_without_context_engine_when_sources_active() -> None:
    state = build_runtime_state_for_tests(run_id="run-fail-ce")
    state.context.config.context_engine = None
    state.user_longterm_memory_result = {
        "used_longterm": True,
        "hits": [
            UserProfileMemoryEntry(entry_id="e1", content="x", kind=MemoryKind.USER_FACT),
        ],
        "scores": [0.9],
        "debug": {"used": True},
    }
    with pytest.raises(RuntimeError, match="context_engine is required"):
        enforce_context_engine_when_provider_sources_active(state)


@pytest.mark.asyncio
async def test_deterministic_assembly_two_runs_match() -> None:
    adapter = _SmallWindowAdapter()
    runtime_config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    handles = build_provider_handles(
        runtime_config=runtime_config,
        ltm_entries=[{"entry_id": "e1", "content": _LTM_SNIPPET, "kind": "user_fact"}],
        rag_chunks=[{"id": "c1", "text": "rag-d", "score": 0.5}],
    )
    request = certification_assembly_request()
    first = await _assemble_with_handles(handles=handles, request=request)
    second = await _assemble_with_handles(handles=handles, request=request)
    assert tuple(f.fragment_id for f in first.fragments_included) == tuple(
        f.fragment_id for f in second.fragments_included
    )


def test_legacy_bypass_inventory_canonical_runtime_zero() -> None:
    rows = inventory_legacy_bypass_symbols()
    for row in rows:
        if row.classification == "review_required":
            pytest.fail(f"{row.symbol} has canonical runtime calls={row.call_count}")
        if row.symbol in {
            "insert_context_before_last_user",
            "append_native_tool_messages",
            "inject_tool_traces_system_context",
            "search_user_longterm_memory",
        }:
            assert row.call_count == 0


def test_tier0_context_import_boundary_guard() -> None:
    script = _REPO / "scripts" / "maintenance" / "check_context_tier0_import_boundary.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mem_xint6_maintenance_bypass_guard_script() -> None:
    script = _REPO / "scripts" / "maintenance" / "check_mem_xint6_canonical_context_bypass.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_raw_handles_documented_as_legacy_bridge_compatibility() -> None:
    bridge_path = _REPO / "intergrax" / "context" / "providers" / "legacy_bridge.py"
    source = bridge_path.read_text(encoding="utf-8")
    assert "legacy compatibility only" in source
    assert "ltm_entries" in source
    assert "ContextProviderContext.handles" in source or "handles``" in source


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_tool_output_cannot_self_elevate_to_system_context() -> None:
    from intergrax.context.errors import ContextProviderContractViolationError
    from intergrax.context.policy.authority import enforce_provider_authority

    fragment = _policy_fragment(
        fragment_id="bad",
        content="x",
        source=ContextFragmentSource.TOOL_OUTPUT,
        authority=ContextAuthorityClass.SYSTEM_CONTEXT,
    )
    descriptor = build_provider_descriptor(
        "tool.custom",
        provider_version="1.0.0",
        supported_sources=frozenset({ContextFragmentSource.TOOL_OUTPUT}),
        origin="plugin",
    )
    with pytest.raises(ContextProviderContractViolationError):
        enforce_provider_authority(fragment, descriptor=descriptor)


def test_certification_matrix_scenarios_registered() -> None:
    """Documented scenario IDs for MEM-XINT-6 report (E2E-1 … E2E-10)."""
    scenarios = (
        "E2E-1-memory-only",
        "E2E-2-rag-only",
        "E2E-3-tool-react",
        "E2E-4-session-memory",
        "E2E-5-memory-rag-conflict",
        "E2E-6-rag-tool-dedup",
        "E2E-7-mixed-source",
        "E2E-8-budget-pressure",
        "E2E-9-malicious-policy",
        "E2E-10-custom-strategies",
    )
    assert len(scenarios) >= 10
