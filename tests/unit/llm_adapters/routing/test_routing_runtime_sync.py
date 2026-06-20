# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext
from intergrax.llm_adapters.routing.context_bridge import LLMRoutingRuntimeSnapshot
from intergrax.llm_adapters.routing.runtime_sync import refresh_config_routing_snapshot
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from testing_support.builder import FakeLLMAdapter


@pytest.mark.unit
@pytest.mark.gate
def test_refresh_config_routing_snapshot_uses_adapter_token_metering() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    inner = FakeLLMAdapter(fixed_text="ok")
    inner.model = "gpt-4o-mini"
    adapter = RoutingEvaluatingLLMAdapter(
        env=env,
        inner=inner,
        context_provider=lambda: RoutingContext(),
        adapter_factory=lambda _eval, _ctx: inner,
    )
    inner.generate_messages([ChatMessage(role="user", content="seed")], run_id="run-sync")

    config = RuntimeConfig(
        llm_adapter=adapter,
        production_mode=False,
        llm_routing_snapshot=LLMRoutingRuntimeSnapshot(),
        run_budget=RunBudget(max_total_tokens=1000),
    )
    context = refresh_config_routing_snapshot(config, run_id="run-sync")
    assert context is not None
    assert config.llm_routing_snapshot is not None
    assert config.llm_routing_snapshot.budget_limits is not None
    assert config.llm_routing_snapshot.budget_limits.agent_tokens_remaining < 1000
