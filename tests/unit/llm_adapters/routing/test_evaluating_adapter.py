# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    RoutingContext,
    RoutingEvaluatingLLMAdapter,
)
from intergrax.llm.messages import ChatMessage
from testing_support.builder import FakeLLMAdapter


@pytest.mark.unit
@pytest.mark.gate
def test_evaluating_adapter_swaps_inner_on_budget_change(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _fake_create(_env: object, evaluation: object) -> FakeLLMAdapter:
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        assert isinstance(evaluation, RoutingEvaluation)
        if evaluation.selected_profile.model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    monkeypatch.setattr(
        "intergrax.llm_adapters.routing.evaluating_adapter.create_adapter_for_routing_evaluation",
        _fake_create,
        raising=False,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.create_adapter_for_routing_evaluation",
        _fake_create,
    )

    ratio_holder = {"ratio": 0.9}

    def _provider() -> RoutingContext:
        return RoutingContext(budget_remaining_ratio=ratio_holder["ratio"])

    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner_primary,
        context_provider=_provider,
    )
    adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert adapter.model == "gpt-4o-mini"

    ratio_holder["ratio"] = 0.1
    adapter.generate_messages([ChatMessage(role="user", content="hi")])
    assert adapter.model == "meta-llama/Llama-3.1-8B"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluating_adapter_emits_on_evaluated_callback() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(default_profile=primary, allowed_profiles=(primary,))
    inner = FakeLLMAdapter()
    seen: list[str] = []

    def _on_evaluated(evaluation: object) -> None:
        from intergrax.llm_adapters.routing.contracts import RoutingEvaluation

        assert isinstance(evaluation, RoutingEvaluation)
        seen.append(evaluation.routing_reason)

    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner,
        context_provider=lambda: RoutingContext(),
        on_evaluated=_on_evaluated,
    )
    adapter.generate_messages([ChatMessage(role="user", content="ping")])
    assert len(seen) == 1
    assert seen[0] == "default_profile"
