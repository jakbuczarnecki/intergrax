# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext
from intergrax.llm_adapters.routing.metering import resolve_metering_adapter, tokens_used_from_adapter
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from testing_support.builder import FakeLLMAdapter


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_metering_adapter_unwraps_evaluating_wrapper() -> None:
    inner = FakeLLMAdapter()
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=env.llm_profile,
        allowed_profiles=(env.llm_profile,),
    )
    wrapper = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner,
        context_provider=lambda: RoutingContext(),
        adapter_factory=lambda _eval, _ctx: inner,
    )
    assert resolve_metering_adapter(wrapper) is inner


@pytest.mark.unit
@pytest.mark.gate
def test_tokens_used_from_adapter_reads_inner_usage() -> None:
    inner = FakeLLMAdapter(fixed_text="ok")
    inner.generate_messages([ChatMessage(role="user", content="hi")], run_id="run-1")
    assert tokens_used_from_adapter(inner, run_id="run-1") > 0


@pytest.mark.unit
@pytest.mark.gate
def test_usage_tracker_registers_inner_after_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    inner_primary = FakeLLMAdapter(fixed_text="primary")
    inner_primary.model = "gpt-4o-mini"
    inner_local = FakeLLMAdapter(fixed_text="local")
    inner_local.model = "meta-llama/Llama-3.1-8B"

    def _factory(
        _env: object,
        evaluation: object,
        _ctx: object | None = None,
    ) -> FakeLLMAdapter:
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        assert isinstance(evaluation, RoutingEvaluation)
        if evaluation.selected_profile.model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.create_adapter_for_routing_evaluation",
        _factory,
    )

    ratio_holder = {"ratio": 0.9}
    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner_primary,
        context_provider=lambda: RoutingContext(budget_remaining_ratio=ratio_holder["ratio"]),
    )
    tracker = LLMUsageTracker(run_id="run-meter")
    tracker.register_adapter(adapter.inner_adapter, label="core_adapter")
    adapter.set_on_inner_swapped(
        lambda inner: tracker.register_adapter(inner, label=f"core_inner_{id(inner)}"),
    )

    adapter.generate_messages([ChatMessage(role="user", content="one")], run_id="run-meter")
    ratio_holder["ratio"] = 0.1
    adapter.generate_messages([ChatMessage(role="user", content="two")], run_id="run-meter")

    report = tracker.build_report()
    assert report.total.calls == 2
    assert report.total.total_tokens > 0
