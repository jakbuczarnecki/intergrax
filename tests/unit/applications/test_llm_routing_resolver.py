# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.llm_resolver import evaluate_llm_routing, resolve_llm_routing_decision
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_llm_routing_uses_profile_rules() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.25, profile=local),),
    )
    selected, hint, reason = evaluate_llm_routing(
        env,
        routing_context=RoutingContext(budget_remaining_ratio=0.1),
    )
    assert selected.model == "meta-llama/Llama-3.1-8B"
    assert reason is not None
    assert reason.startswith("rule:")
    decision = resolve_llm_routing_decision(
        env,
        routing_context=RoutingContext(budget_remaining_ratio=0.1),
    )
    assert decision.model == "meta-llama/Llama-3.1-8B"
