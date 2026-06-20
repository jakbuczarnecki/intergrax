# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.llm_adapters.routing import (
    BudgetBelowRule,
    LLMRoutingProfile,
    RoutingContext,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from testing_support.builder import FakeLLMAdapter


@pytest.mark.integration
@pytest.mark.gate
def test_mid_run_budget_threshold_swaps_adapter_model(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _fake_create(
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
        _fake_create,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver._create_base_llm_adapter",
        lambda _env, _profile, hint=None: inner_primary,
    )

    ratio_holder = {"ratio": 0.9}

    adapter = resolve_llm_adapter(
        env,
        routing_context=RoutingContext(budget_remaining_ratio=0.9),
        context_provider=lambda: RoutingContext(budget_remaining_ratio=ratio_holder["ratio"]),
    )
    assert isinstance(adapter, RoutingEvaluatingLLMAdapter)
    adapter.generate_messages([ChatMessage(role="user", content="one")])
    assert adapter.model == "gpt-4o-mini"

    ratio_holder["ratio"] = 0.1
    adapter.generate_messages([ChatMessage(role="user", content="two")])
    assert adapter.model == "meta-llama/Llama-3.1-8B"


@pytest.mark.integration
@pytest.mark.gate
def test_materialize_runtime_config_wraps_evaluating_adapter() -> None:
    env = build_lab_environment_profile(LabApplicationSettings())
    request = RuntimeRequest(
        agent_id="lab-agent",
        user_id="user-1",
        session_id="sess-1",
        tenant_id="lab-tenant",
        message="hello",
        metadata={"task_class": "lab_routing", "agent_id": "lab-agent"},
    )
    config = materialize_runtime_config(request, default_reference_harness(), env)
    assert config.llm_routing_snapshot is not None
    assert isinstance(config.llm_adapter, RoutingEvaluatingLLMAdapter)
