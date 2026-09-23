# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect

import pytest

from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    RoutingContext,
    routing_evaluation_identity,
)
from intergrax.llm_adapters.routing.metering import resolve_metering_adapter, tokens_used_from_adapter
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from testing_support.builder import FakeLLMAdapter


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_state_swap_registration_uses_semantic_route_identity() -> None:
    from intergrax.runtime.nexus.engine import runtime_state as runtime_state_mod

    source = inspect.getsource(runtime_state_mod.RuntimeState.configure_llm_tracker)
    assert "id(inner)" not in source
    assert "routing_evaluation_identity" in source
    assert "core_inner:" in source


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
    swapped_labels: list[str] = []

    def _on_inner_swapped(inner: object, evaluation: object) -> None:
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        assert isinstance(evaluation, RoutingEvaluation)
        label = f"core_inner:{routing_evaluation_identity(evaluation)}"
        swapped_labels.append(label)
        tracker.register_adapter(inner, label=label)

    adapter.set_on_inner_swapped(_on_inner_swapped)

    adapter.generate_messages([ChatMessage(role="user", content="one")], run_id="run-meter")
    ratio_holder["ratio"] = 0.1
    adapter.generate_messages([ChatMessage(role="user", content="two")], run_id="run-meter")

    report = tracker.build_report()
    assert report.total.calls == 2
    assert report.total.total_tokens > 0
    assert swapped_labels
    assert all(label.startswith("core_inner:") for label in swapped_labels)
    assert all("id(" not in label for label in swapped_labels)


@pytest.mark.unit
@pytest.mark.gate
def test_routing_evaluation_identity_is_deterministic_across_instances() -> None:
    from intergrax.llm_adapters.routing.contracts import RoutingEvaluation
    from intergrax.llm_adapters.routing.evaluator import LLMRoutingEvaluator

    profile = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    routing_profile = LLMRoutingProfile(
        default_profile=profile,
        allowed_profiles=(profile,),
    )
    evaluation = LLMRoutingEvaluator().evaluate(routing_profile, RoutingContext())
    assert isinstance(evaluation, RoutingEvaluation)

    identity_a = routing_evaluation_identity(evaluation)
    identity_b = routing_evaluation_identity(evaluation)
    assert identity_a == identity_b
    assert identity_a == "openai:gpt-4o-mini:"


@pytest.mark.unit
@pytest.mark.gate
def test_routing_evaluation_identity_distinguishes_different_routes() -> None:
    from intergrax.llm_adapters.routing.contracts import RoutingEvaluation, RoutingTarget

    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    eval_primary = RoutingEvaluation(
        matched_rule_id=None,
        target=RoutingTarget(profile=primary),
        routing_reason="default",
        selected_profile=primary,
    )
    eval_local = RoutingEvaluation(
        matched_rule_id="budget",
        target=RoutingTarget(profile=local),
        routing_reason="budget",
        selected_profile=local,
    )
    assert routing_evaluation_identity(eval_primary) != routing_evaluation_identity(eval_local)


@pytest.mark.unit
@pytest.mark.gate
def test_usage_tracker_reregister_same_semantic_label_updates_trackable() -> None:
    label = "core_inner:vllm:meta-llama/Llama-3.1-8B:"
    first = FakeLLMAdapter(fixed_text="local-v1")
    first.model = "meta-llama/Llama-3.1-8B"
    second = FakeLLMAdapter(fixed_text="local-v2")
    second.model = "meta-llama/Llama-3.1-8B"
    tracker = LLMUsageTracker(run_id="run-reregister")
    tracker.register_adapter(first, label=label)
    first.generate_messages([ChatMessage(role="user", content="one")], run_id="run-reregister")
    tracker.register_adapter(second, label=label)
    second.generate_messages([ChatMessage(role="user", content="two")], run_id="run-reregister")

    report = tracker.build_report()
    by_label = {entry.label: entry for entry in report.entries}
    assert by_label[label].adapter_instance_id == id(second)
    assert by_label[label].stats.calls == 1
