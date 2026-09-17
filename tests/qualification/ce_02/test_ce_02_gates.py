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
    resolve_authoritative_model_budget,
)
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
from intergrax.context.policy.budget_allocator import DefaultContextBudgetAllocator
from intergrax.context.registry import ContextPluginRegistry
from intergrax.llm.messages import ChatMessage
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


class _FakeAdapter:
    provider = "fake"
    model = "fake-budget"

    def __init__(self, window: int = 8192) -> None:
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def count_messages_tokens(self, messages: object) -> int:
        total = 0
        for message in messages:
            total += max(1, len(getattr(message, "content", "") or "") // 4)
        return total


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


def test_ce2_q4_custom_token_counter_injection() -> None:
    registry = ContextPluginRegistry()

    class _DoubleCounter(CharEstimateContextTokenCounter):
        @property
        def strategy_id(self) -> str:
            return "double_char_estimate.v1"

        def count_text(self, text: str) -> int:
            return super().count_text(text) * 2

    registry.set_token_counter(_DoubleCounter())
    assert registry.token_counter is not None
    assert registry.token_counter.strategy_id == "double_char_estimate.v1"
    assert registry.token_counter.count_text("abcd") == 2


def test_ce2_q5_custom_model_budget_policy() -> None:
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

    capability = ModelContextCapabilitySnapshot(8000, 1000, 256)
    resolved = resolve_authoritative_model_budget(
        capability=capability,
        request=_request(4000),
        policy=_TightPolicy(),
    )
    assert resolved.policy_id == "tight_test_policy"
    assert resolved.available_input_tokens == 128


def test_ce2_q6_custom_compaction_strategy() -> None:
    registry = ContextPluginRegistry()
    strategy = DeterministicTailCompactionStrategy()
    registry.set_compaction_strategy(strategy)
    fragment = ContextFragment(
        fragment_id="f1",
        source=ContextFragmentSource.RAG,
        source_id="s1",
        content="x" * 400,
        token_estimate=100,
        relevance_score=0.5,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
    )
    result = registry.compaction_strategy.compact(
        ContextCompactionInput(fragment=fragment, target_token_budget=10),
    )
    assert result is not None
    assert result.provenance.strategy_id == strategy.strategy_id


def test_ce2_q7_custom_degradation_policy() -> None:
    class _SingleStepPolicy:
        @property
        def policy_id(self) -> str:
            return "single_step_test"

        def ladder_order(self) -> tuple[DegradationStepKind, ...]:
            return (DegradationStepKind.FULL, DegradationStepKind.DROP_LOWEST_SCORED)

    compiler = ContextCompiler(degradation_policy=_SingleStepPolicy())
    assert compiler._degradation_policy.policy_id == "single_step_test"


def test_ce2_q8_mandatory_fragment_preserved() -> None:
    allocator = DefaultContextBudgetAllocator()
    mandatory = ContextFragment(
        fragment_id="m1",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        source_id="sys",
        content="required system",
        token_estimate=500,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    optional = ContextFragment(
        fragment_id="o1",
        source=ContextFragmentSource.RAG,
        source_id="r1",
        content="optional rag",
        token_estimate=500,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    result = allocator.allocate([mandatory, optional], 550, _request())
    included_ids = {f.fragment_id for f in result.included}
    assert "m1" in included_ids
    assert "o1" not in included_ids


def test_ce2_q9_unsatisfiable_mandatory_budget() -> None:
    err = ContextBudgetUnsatisfiableError(mandatory_tokens=900, available_tokens=100)
    assert err.reason_code == "budget.unsatisfiable.mandatory_overflow"
    with pytest.raises(ContextBudgetUnsatisfiableError):
        raise err


def test_ce2_q10_compaction_preserves_governance_fields() -> None:
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
    )
    result = DeterministicTailCompactionStrategy().compact(
        ContextCompactionInput(fragment=fragment, target_token_budget=5),
    )
    assert result is not None
    assert result.fragment.authority_class is ContextAuthorityClass.RAG_EVIDENCE


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
    adapter = _FakeAdapter(window=512)
    config = RuntimeConfig(llm_adapter=adapter)
    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="u" * 4000),
    ]
    compiler = ContextCompiler()
    result = compiler.compile(messages, config, max_output_tokens=64)
    allowed = resolve_input_budget_tokens(adapter, max_output_tokens=64, margin_tokens=compiler._margin_tokens)
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


def test_ce2_q14_deterministic_budget_replay() -> None:
    capability = ModelContextCapabilitySnapshot(12000, 1500, 256)
    request = _request(3500)
    first = resolve_authoritative_model_budget(capability=capability, request=request)
    second = resolve_authoritative_model_budget(capability=capability, request=request)
    assert first == second


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
