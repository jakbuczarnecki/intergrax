# © Artur Czarnecki. All rights reserved.

"""CE-02-R1 budget governance unit tests."""

from __future__ import annotations

import pytest

from intergrax.context.budget import (
    ContextBudgetResolveInput,
    ContextBudgetUnsatisfiableError,
    DefaultContextModelBudgetPolicy,
    ModelContextCapabilitySnapshot,
    global_allocatable_tokens,
    resolve_authoritative_model_budget,
)
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.policy.budget_allocator import DefaultContextBudgetAllocator
from intergrax.context.policy.pipeline import ContextCrossSourcePolicyPipeline
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.runtime.nexus.context.context_preflight import verify_context_preflight
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Adapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-ce02r1"

    def __init__(self, window: int = 512) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def _assembly_request(max_tokens: int) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-ce02r1a",
        run_id="run-ce02r1a",
        task_id="task-ce02r1a",
        tenant_id="tenant-a",
        assembly_scope="graph_node",
        objective="mandatory budget accounting",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=max_tokens),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="model_call",
    )


def test_mandatory_fragments_do_not_consume_allocatable_budget() -> None:
    """Regression: allocatable_tokens is optional-only (CE-02-R1A audit)."""
    mandatory = ContextFragment(
        fragment_id="mandatory-300",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        source_id="sys",
        content="required",
        token_estimate=300,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    optional_fragments = [
        ContextFragment(
            fragment_id=f"opt-{index}",
            source=ContextFragmentSource.RAG,
            source_id=f"doc-{index}",
            content=f"optional-{index}",
            token_estimate=200,
            relevance_score=0.95,
            freshness_score=0.9,
            confidence_score=0.9,
            mandatory=False,
        )
        for index in range(3)
    ]
    allocatable = 600
    pipeline = ContextCrossSourcePolicyPipeline()
    result = pipeline.execute(
        [mandatory, *optional_fragments],
        _assembly_request(1000),
        fragment_budget_tokens=allocatable,
    )
    included_ids = {fragment.fragment_id for fragment in result.fragments}
    assert "mandatory-300" in included_ids
    assert included_ids >= {"opt-0", "opt-1", "opt-2"}

    allocator_only = DefaultContextBudgetAllocator().allocate(
        [mandatory, *optional_fragments],
        allocatable,
        _assembly_request(1000),
    )
    allocator_ids = {fragment.fragment_id for fragment in allocator_only.included}
    assert allocator_ids >= {"opt-0", "opt-1", "opt-2"}


def test_global_allocatable_tokens_preserves_zero() -> None:
    capability = ModelContextCapabilitySnapshot(8000, 1000, 256)
    reserve = capability.available_input_tokens
    resolved = DefaultContextModelBudgetPolicy().resolve_budget(
        ContextBudgetResolveInput(
            capability=capability,
            request_budget=ContextBudgetSnapshot(max_tokens_estimate=reserve),
            mandatory_reserve_tokens=reserve,
        ),
    )
    assert resolved.allocatable_tokens == 0
    assert global_allocatable_tokens(resolved) == 0


def test_mandatory_reserve_overflow_raises() -> None:
    from intergrax.context.contracts import ContextAssemblyRequest, ContextDecisionSnapshot
    from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

    capability = ModelContextCapabilitySnapshot(8000, 1000, 256)
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="acp_step",
        objective="overflow",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=4000),
        assembly_options=TaskContextAssemblyOptions(),
    )
    with pytest.raises(ContextBudgetUnsatisfiableError):
        resolve_authoritative_model_budget(
            capability=capability,
            request=request,
            mandatory_reserve_tokens=capability.available_input_tokens + 1,
        )


def test_compiler_total_tokens_matches_message_count() -> None:
    adapter = _Adapter(window=4096)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello world"),
    ]
    compiler = ContextCompiler()
    result = compiler.compile(messages, config, max_output_tokens=64, input_budget_tokens=500)
    actual = sum(compiler.count_tokens(message.content or "") for message in result.messages)
    assert result.total_tokens == actual


def test_compiler_success_implies_preflight_success() -> None:
    adapter = _Adapter(window=4096)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    messages = [ChatMessage(role="user", content="short")]
    compiler = ContextCompiler()
    result = compiler.compile(messages, config, max_output_tokens=64, input_budget_tokens=200)
    verify_context_preflight(
        result.messages,
        adapter,
        max_output_tokens=64,
        count_tokens=compiler.count_tokens,
    )


def test_oversized_mandatory_user_turn_fails_closed() -> None:
    adapter = _Adapter(window=512)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    messages = [
        ChatMessage(role="system", content="s"),
        ChatMessage(role="user", content="u" * 4000),
    ]
    compiler = ContextCompiler()
    with pytest.raises(ContextBudgetUnsatisfiableError):
        compiler.compile(messages, config, max_output_tokens=64)
