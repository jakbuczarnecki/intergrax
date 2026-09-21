# © Artur Czarnecki. All rights reserved.

"""CE-02 qualification gates (CE2-Q1..CE2-Q18 primary evidence)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.context.budget import (
    CharEstimateContextTokenCounter,
    ContextBudgetResolveInput,
    ContextCompactionInput,
    ContextBudgetUnsatisfiableError,
    DefaultContextModelBudgetPolicy,
    DeterministicTailCompactionStrategy,
    ModelContextCapabilitySnapshot,
    ResolvedModelContextBudget,
    global_allocatable_tokens,
    resolve_authoritative_model_budget,
)
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.contracts import ContextProviderContext
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
)
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.context.budget.degradation import DefaultContextDegradationPolicy
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextAuthorityClass,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_budget import resolve_input_budget_tokens
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_compiler_models import DegradationStepKind
from intergrax.runtime.nexus.context.context_engine import (
    DefaultNexusContextEngine,
    _compile_preserved_planned_context,
)
from intergrax.runtime.nexus.context.model_capability import snapshot_model_capability
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BUDGET_TIER0 = _REPO_ROOT / "intergrax" / "context" / "budget"
_FORBIDDEN_VENDOR_PREFIXES = (
    "openai",
    "anthropic",
    "google.genai",
    "google.generativeai",
    "boto3",
    "azure.ai",
    "bedrock",
)


class _FakeAdapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-budget"

    def __init__(self, window: int = 8192) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")

    def count_messages_tokens(self, messages: object) -> int:
        total = 0
        for message in messages:
            total += max(1, len(message.content or "") // 4)
        return total


class _MandatoryWebsearchOptionalProvider:
    provider_id = "test.mandatory_websearch_optional"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset(
            {
                ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                ContextFragmentSource.WEBSEARCH,
            }
        )

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self.supported_sources,
            origin="test",
        )

    async def collect(self, request: ContextAssemblyRequest, ctx: ContextProviderContext) -> list[ContextFragment]:
        _ = request, ctx
        return [
            ContextFragment(
                fragment_id="ce2-q7-mandatory",
                source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                source_id="policy",
                content="CE2-Q7-MANDATORY-MARKER",
                token_estimate=40,
                relevance_score=1.0,
                freshness_score=1.0,
                confidence_score=1.0,
                mandatory=True,
            ),
            ContextFragment(
                fragment_id="ce2-q7-optional-websearch",
                source=ContextFragmentSource.WEBSEARCH,
                source_id="web-opt",
                content="WEBSEARCH:\n" + ("w" * 1200),
                token_estimate=10,
                relevance_score=0.95,
                freshness_score=0.95,
                confidence_score=0.95,
                mandatory=False,
            ),
        ]


class _MandatoryOptionalProvider:
    provider_id = "test.mandatory_optional"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset(
            {
                ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                ContextFragmentSource.RAG,
            }
        )

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self.supported_sources,
            origin="test",
        )

    async def collect(self, request: ContextAssemblyRequest, ctx: ContextProviderContext) -> list[ContextFragment]:
        _ = request, ctx
        return [
            ContextFragment(
                fragment_id="ce2-q8-mandatory",
                source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
                source_id="policy",
                content="CE2-Q8-MANDATORY-MARKER",
                token_estimate=80,
                relevance_score=1.0,
                freshness_score=1.0,
                confidence_score=1.0,
                mandatory=True,
            ),
            ContextFragment(
                fragment_id="ce2-q8-optional",
                source=ContextFragmentSource.RAG,
                source_id="doc-opt",
                content="CE2-Q8-OPTIONAL-MARKER " + ("o" * 1200),
                token_estimate=700,
                relevance_score=0.2,
                freshness_score=0.2,
                confidence_score=0.2,
                mandatory=False,
            ),
        ]


class _RagOverflowProvider:
    provider_id = "test.rag_overflow"

    @property
    def supported_sources(self) -> frozenset[ContextFragmentSource]:
        return frozenset({ContextFragmentSource.RAG})

    @property
    def descriptor(self):
        return build_provider_descriptor(
            self.provider_id,
            provider_version="1.0.0",
            supported_sources=self.supported_sources,
            origin="test",
        )

    async def collect(self, request: ContextAssemblyRequest, ctx: ContextProviderContext) -> list[ContextFragment]:
        _ = request, ctx
        return [
            ContextFragment(
                fragment_id="rag-big",
                source=ContextFragmentSource.RAG,
                source_id="doc-1",
                content="r" * 800,
                token_estimate=10,
                relevance_score=0.95,
                freshness_score=0.95,
                confidence_score=0.95,
                mandatory=False,
            )
        ]


def _assemble_runtime(adapter: LLMAdapter, content: str, *, max_output: int = 64):
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=RuntimeConfig(llm_adapter=adapter, production_mode=False),
        messages=[ChatMessage(role="user", content=content)],
        max_output_tokens=max_output,
    )
    return ContextProviderContext(engine_id="ce02", runtime=runtime)


def _request(budget_tokens: int = 4000) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-budget",
        run_id="r-budget",
        task_id="t-budget",
        tenant_id="tenant-a",
        assembly_scope="graph_node",
        objective="ce-02 gate",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=budget_tokens),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="model_call",
    )


def test_ce2_q1_single_budget_resolution_entry() -> None:
    source = inspect.getsource(resolve_authoritative_model_budget)
    assert "resolve_budget" in source
    engine_source = inspect.getsource(DefaultNexusContextEngine._assemble_inner)
    assert "resolve_authoritative_model_budget" in engine_source
    assert "fragment_budget_tokens" in engine_source


def test_ce2_q2_typed_resolved_budget_contract() -> None:
    capability = ModelContextCapabilitySnapshot(
        model_context_window=10000,
        reserved_output_tokens=2000,
        platform_margin_tokens=256,
    )
    resolved = DefaultContextModelBudgetPolicy().resolve_budget(
        ContextBudgetResolveInput(capability=capability, request_budget=ContextBudgetSnapshot(max_tokens_estimate=3000)),
    )
    assert isinstance(resolved, ResolvedModelContextBudget)
    assert resolved.available_input_tokens <= capability.available_input_tokens
    assert resolved.policy_id


def test_ce2_q3_capability_snapshot_no_vendor_types_in_tier0() -> None:
    for path in _BUDGET_TIER0.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in _FORBIDDEN_VENDOR_PREFIXES:
                        assert not alias.name.startswith(prefix), alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in _FORBIDDEN_VENDOR_PREFIXES:
                    assert not node.module.startswith(prefix), node.module


@pytest.mark.asyncio
async def test_ce2_q4_custom_token_counter_injection() -> None:
    registry = ContextPluginRegistry()

    class _DoubleCounter(CharEstimateContextTokenCounter):
        @property
        def strategy_id(self) -> str:
            return "double_char_estimate.v1"

        def count_text(self, text: str) -> int:
            return super().count_text(text) * 2

    registry.set_token_counter(_DoubleCounter())
    adapter = _FakeAdapter(window=4096)
    engine = DefaultNexusContextEngine(registry=registry)
    ctx = _assemble_runtime(adapter, "x" * 40)
    assembled = await engine.assemble(_request(500), provider_ctx=ctx)
    assert assembled.total_tokens == _DoubleCounter().count_messages(assembled.messages)


@pytest.mark.asyncio
async def test_ce2_q5_custom_model_budget_policy() -> None:
    class _TightPolicy(DefaultContextModelBudgetPolicy):
        @property
        def policy_id(self) -> str:
            return "tight_test_policy"

        def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget:
            base = super().resolve_budget(inputs)
            return ResolvedModelContextBudget(
                model_context_window=base.model_context_window,
                reserved_output_tokens=base.reserved_output_tokens,
                platform_margin_tokens=base.platform_margin_tokens,
                available_input_tokens=min(base.available_input_tokens, 128),
                mandatory_reserve_tokens=base.mandatory_reserve_tokens,
                allocatable_tokens=min(base.allocatable_tokens, 128),
                request_cap_tokens=base.request_cap_tokens,
                policy_id=self.policy_id,
                policy_version="test",
            )

    registry = ContextPluginRegistry()
    registry.set_model_budget_policy(_TightPolicy())
    adapter = _FakeAdapter(window=8000)
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(
        _request(4000),
        provider_ctx=_assemble_runtime(adapter, "hi"),
    )
    assert assembled.resolved_model_budget is not None
    assert assembled.resolved_model_budget.policy_id == "tight_test_policy"
    assert assembled.resolved_model_budget.available_input_tokens == 128


@pytest.mark.asyncio
async def test_ce2_q6_custom_compaction_strategy() -> None:
    class _TightAllocPolicy(DefaultContextModelBudgetPolicy):
        @property
        def policy_id(self) -> str:
            return "tight_alloc_for_compaction"

        def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget:
            base = super().resolve_budget(inputs)
            return ResolvedModelContextBudget(
                model_context_window=base.model_context_window,
                reserved_output_tokens=base.reserved_output_tokens,
                platform_margin_tokens=base.platform_margin_tokens,
                available_input_tokens=48,
                mandatory_reserve_tokens=base.mandatory_reserve_tokens,
                allocatable_tokens=max(0, 24 - base.mandatory_reserve_tokens),
                request_cap_tokens=base.request_cap_tokens,
                policy_id=self.policy_id,
                policy_version="test",
            )

    registry = ContextPluginRegistry()
    strategy = DeterministicTailCompactionStrategy()
    registry.set_compaction_strategy(strategy)
    registry.set_model_budget_policy(_TightAllocPolicy())
    registry.add_provider(_RagOverflowProvider())
    adapter = _FakeAdapter(window=4096)
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(
        _request(48),
        provider_ctx=_assemble_runtime(adapter, "user question"),
    )
    assert assembled.compaction_strategy_id == strategy.strategy_id
    assert assembled.compaction_provenance
    assert assembled.compaction_provenance[0].strategy_id == strategy.strategy_id


@pytest.mark.asyncio
async def test_ce2_q7_custom_degradation_policy() -> None:
    class _CustomDegradationPolicy(DefaultContextDegradationPolicy):
        @property
        def policy_id(self) -> str:
            return "single_step_test"

        def ladder_order(self) -> tuple[DegradationStepKind, ...]:
            return (
                DegradationStepKind.FULL,
                DegradationStepKind.DROP_OPTIONAL_INJECTIONS,
                DegradationStepKind.DROP_LOWEST_SCORED,
            )

    class _TightPolicy(DefaultContextModelBudgetPolicy):
        @property
        def policy_id(self) -> str:
            return "tight_for_degradation"

        def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget:
            base = super().resolve_budget(inputs)
            return ResolvedModelContextBudget(
                model_context_window=base.model_context_window,
                reserved_output_tokens=base.reserved_output_tokens,
                platform_margin_tokens=base.platform_margin_tokens,
                available_input_tokens=280,
                mandatory_reserve_tokens=base.mandatory_reserve_tokens,
                allocatable_tokens=max(0, 260 - base.mandatory_reserve_tokens),
                request_cap_tokens=base.request_cap_tokens,
                policy_id=self.policy_id,
                policy_version="test",
            )

    registry = ContextPluginRegistry()
    registry.set_degradation_policy(_CustomDegradationPolicy())
    registry.set_model_budget_policy(_TightPolicy())
    registry.add_provider(_MandatoryWebsearchOptionalProvider())
    adapter = _FakeAdapter(window=4096)
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(
        _request(280),
        provider_ctx=_assemble_runtime(adapter, "hi"),
    )
    assert assembled.degradation_policy_id == "single_step_test"
    assert assembled.total_tokens <= assembled.budget_tokens
    assert DegradationStepKind.DROP_OPTIONAL_INJECTIONS.value in assembled.degradation_steps
    assert DegradationStepKind.FULL.value not in assembled.degradation_steps
    included_ids = {fragment.fragment_id for fragment in assembled.fragments_included}
    excluded_ids = {fragment.fragment_id for fragment, _reason in assembled.fragments_excluded}
    provenance_fragment_ids = {item.fragment_id for item in assembled.provenance}
    assert "ce2-q7-mandatory" in included_ids
    assert "ce2-q7-optional-websearch" in excluded_ids
    assert "ce2-q7-optional-websearch" not in included_ids
    assert "ce2-q7-optional-websearch" not in provenance_fragment_ids
    model_facing_text = "\n".join(message.content or "" for message in assembled.messages)
    assert "CE2-Q7-MANDATORY-MARKER" in model_facing_text
    assert "ce2-q7-optional-websearch" not in model_facing_text
    assert "WEBSEARCH" not in model_facing_text


@pytest.mark.asyncio
async def test_ce2_q8_mandatory_fragment_preserved() -> None:
    class _TightPolicy(DefaultContextModelBudgetPolicy):
        @property
        def policy_id(self) -> str:
            return "tight_for_mandatory_e2e"

        def resolve_budget(self, inputs: ContextBudgetResolveInput) -> ResolvedModelContextBudget:
            base = super().resolve_budget(inputs)
            return ResolvedModelContextBudget(
                model_context_window=base.model_context_window,
                reserved_output_tokens=base.reserved_output_tokens,
                platform_margin_tokens=base.platform_margin_tokens,
                available_input_tokens=160,
                mandatory_reserve_tokens=base.mandatory_reserve_tokens,
                allocatable_tokens=max(0, 60 - base.mandatory_reserve_tokens),
                request_cap_tokens=base.request_cap_tokens,
                policy_id=self.policy_id,
                policy_version="test",
            )

    registry = ContextPluginRegistry()
    registry.set_model_budget_policy(_TightPolicy())
    registry.add_provider(_MandatoryOptionalProvider())
    adapter = _FakeAdapter(window=4096)
    engine = DefaultNexusContextEngine(registry=registry)
    assembled = await engine.assemble(
        _request(160),
        provider_ctx=_assemble_runtime(adapter, "short user turn"),
    )
    model_facing_text = "\n".join(message.content or "" for message in assembled.messages)
    assert "CE2-Q8-MANDATORY-MARKER" in model_facing_text
    assert "CE2-Q8-OPTIONAL-MARKER" not in model_facing_text
    included_ids = {fragment.fragment_id for fragment in assembled.fragments_included}
    assert "ce2-q8-mandatory" in included_ids
    excluded_ids = {fragment.fragment_id for fragment, _reason in assembled.fragments_excluded}
    assert "ce2-q8-optional" in excluded_ids
    provenance_fragment_ids = {item.fragment_id for item in assembled.provenance}
    assert "ce2-q8-mandatory" in provenance_fragment_ids
    assert "ce2-q8-optional" not in provenance_fragment_ids


@pytest.mark.asyncio
async def test_ce2_q9_unsatisfiable_mandatory_budget() -> None:
    adapter = _FakeAdapter(window=256)
    engine = DefaultNexusContextEngine()
    huge_user = "m" * 2000
    with pytest.raises(ContextBudgetUnsatisfiableError):
        await engine.assemble(
            _request(200),
            provider_ctx=_assemble_runtime(adapter, huge_user, max_output=32),
        )


def test_ce2_q10_compaction_preserves_governance_fields() -> None:
    from intergrax.contracts.data_classification import DataClassification
    from intergrax.context.contracts import ContextFragmentScopeRef

    scope = ContextFragmentScopeRef(tenant_id="tenant-a", execution_scope_key="task-1")
    fragment = ContextFragment(
        fragment_id="g1",
        source=ContextFragmentSource.RAG,
        source_id="src",
        content="z" * 200,
        token_estimate=50,
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
        authority_class=ContextAuthorityClass.RAG_EVIDENCE,
        sensitivity=DataClassification.CONFIDENTIAL,
        scope_ref=scope,
        provider_provenance=None,
    )
    result = DeterministicTailCompactionStrategy().compact(
        ContextCompactionInput(fragment=fragment, target_token_budget=5),
    )
    assert result is not None
    assert result.fragment.authority_class is ContextAuthorityClass.RAG_EVIDENCE
    assert result.fragment.scope_ref == scope
    assert result.fragment.sensitivity is DataClassification.CONFIDENTIAL
    assert result.fragment.source_id == "src"


def test_ce2_q11_compaction_provenance() -> None:
    fragment = ContextFragment(
        fragment_id="p1",
        source=ContextFragmentSource.TOOL_OUTPUT,
        source_id="tool",
        content="y" * 300,
        token_estimate=75,
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
    )
    result = DeterministicTailCompactionStrategy().compact(
        ContextCompactionInput(fragment=fragment, target_token_budget=8),
    )
    assert result is not None
    assert result.provenance.source_fragment_ids == ("p1",)
    assert result.provenance.reason_code == "budget.compacted.fragment"


def test_ce2_q12_model_window_after_compile() -> None:
    adapter = _FakeAdapter(window=4096)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="short question"),
    ]
    compiler = ContextCompiler()
    allowed = resolve_input_budget_tokens(adapter, max_output_tokens=64, margin_tokens=compiler.margin_tokens)
    result = compiler.compile(messages, config, max_output_tokens=64, input_budget_tokens=allowed)
    actual = sum(compiler.count_tokens(message.content or "") for message in result.messages)
    assert result.total_tokens == actual
    assert result.total_tokens <= allowed


def test_ce2_q13_compile_plan_invariant_guard_present() -> None:
    assert _compile_preserved_planned_context(
        degradation_steps=(DegradationStepKind.FULL.value,),
        planned_hash="abc",
        compiled_hash="abc",
        compiled_budget_tokens=100,
        planned_budget_tokens=100,
    )
    assert not _compile_preserved_planned_context(
        degradation_steps=("drop",),
        planned_hash="abc",
        compiled_hash="def",
        compiled_budget_tokens=100,
        planned_budget_tokens=100,
    )
    engine_source = inspect.getsource(DefaultNexusContextEngine._assemble_inner)
    assert "FINAL_COMPILE_MUTATED_PLAN" in engine_source


@pytest.mark.asyncio
async def test_ce2_q14_deterministic_budget_replay() -> None:
    registry = ContextPluginRegistry()
    registry.add_provider(_RagOverflowProvider())
    adapter = _FakeAdapter(window=4096)
    engine = DefaultNexusContextEngine(registry=registry)
    request = _request(400)
    provider_ctx = _assemble_runtime(adapter, "deterministic-user-turn")
    first = await engine.assemble(request, provider_ctx=provider_ctx)
    second = await engine.assemble(request, provider_ctx=provider_ctx)

    assert tuple(fragment.fragment_id for fragment in first.fragments_included) == tuple(
        fragment.fragment_id for fragment in second.fragments_included
    )
    excluded_key = lambda assembled: tuple(
        (fragment.fragment_id, reason) for fragment, reason in assembled.fragments_excluded
    )
    assert excluded_key(first) == excluded_key(second)
    assert first.compaction_provenance == second.compaction_provenance
    assert first.degradation_steps == second.degradation_steps
    assert first.budget_tokens == second.budget_tokens
    assert first.total_tokens == second.total_tokens
    assert compute_model_facing_messages_hash(first.messages) == compute_model_facing_messages_hash(
        second.messages
    )


def test_ce2_q15_hidden_truncation_gate() -> None:
    tier0 = _REPO_ROOT / "intergrax" / "context"
    suspicious: list[str] = []
    for path in tier0.rglob("*.py"):
        if "budget" not in path.parts and path.name not in {"compaction.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "message[:]" in text.replace(" ", ""):
            suspicious.append(str(path))
    assert not suspicious


def test_ce2_q16_tier0_budget_vendor_import_gate() -> None:
    test_ce2_q3_capability_snapshot_no_vendor_types_in_tier0()


def test_ce2_q17_budget_contract_static_abi() -> None:
    forbidden = ("getattr(", "hasattr(", "service_locator", "GLOBAL_REGISTRY")
    for path in _BUDGET_TIER0.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains {token}"


def test_snapshot_model_capability_matches_window_math() -> None:
    adapter = _FakeAdapter(4096)
    snap = snapshot_model_capability(adapter, max_output_tokens=512, margin_tokens=256)
    assert snap.model_context_window == 4096
    assert snap.available_input_tokens == 4096 - 512 - 256
