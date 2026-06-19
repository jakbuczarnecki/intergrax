# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


@pytest.mark.integration
@pytest.mark.gate
def test_production_metering_mid_run_budget_swap_with_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = build_lab_environment_profile(LabApplicationSettings())
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

    def _fake_create(_env: object, evaluation: object, _ctx: object = None) -> FakeLLMAdapter:
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

    request = RuntimeRequest(
        agent_id="lab-agent",
        user_id="user-1",
        session_id="sess-1",
        tenant_id="lab-tenant",
        message="hello",
        metadata={"task_class": "lab_routing", "agent_id": "lab-agent", "run_id": "run-prod-meter"},
    )
    config = materialize_runtime_config(
        request,
        default_reference_harness(),
        env,
    )
    assert isinstance(config.llm_adapter, RoutingEvaluatingLLMAdapter)

    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext

    runtime_context = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    state = RuntimeState(context=runtime_context, request=request, run_id="run-prod-meter")
    state.configure_llm_tracker()

    ratio_holder = {"ratio": 0.9}
    evaluating = config.llm_adapter
    assert isinstance(evaluating, RoutingEvaluatingLLMAdapter)
    evaluating.set_context_provider(lambda: RoutingContext(budget_remaining_ratio=ratio_holder["ratio"]))

    evaluating.generate_messages([ChatMessage(role="user", content="one")], run_id="run-prod-meter")
    assert evaluating.model == "gpt-4o-mini"

    ratio_holder["ratio"] = 0.1
    evaluating.generate_messages([ChatMessage(role="user", content="two")], run_id="run-prod-meter")
    assert evaluating.model == "meta-llama/Llama-3.1-8B"

    report = state.llm_usage_tracker.build_report() if state.llm_usage_tracker else None
    assert report is not None
    assert report.total.calls == 2
    assert report.total.total_tokens > 0
