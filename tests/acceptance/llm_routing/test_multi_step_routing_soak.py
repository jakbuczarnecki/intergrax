# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.agents.reference_harness import default_reference_harness
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile, RoutingContext
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


@pytest.mark.integration
@pytest.mark.gate
def test_multi_step_routing_soak_budget_burn_and_trace(
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

    def _profile_create(self, **overrides):  # type: ignore[no-untyped-def]
        model = overrides.get("model") or self.model
        if model == "meta-llama/Llama-3.1-8B":
            return inner_local
        return inner_primary

    monkeypatch.setattr(LLMProfile, "create_adapter", _profile_create)
    monkeypatch.setattr(LLMProfile, "create_adapter_with_failover", _profile_create)

    request = RuntimeRequest(
        agent_id="lab-agent",
        user_id="user-1",
        session_id="sess-1",
        tenant_id="lab-tenant",
        message="hello",
        metadata={"task_class": "lab_routing", "agent_id": "lab-agent", "run_id": "run-soak"},
    )
    config = materialize_runtime_config(request, default_reference_harness(), env)
    evaluating = config.llm_adapter
    assert isinstance(evaluating, RoutingEvaluatingLLMAdapter)

    runtime_context = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    state = RuntimeState(context=runtime_context, request=request, run_id="run-soak")
    state.configure_llm_tracker()

    ratios = iter([0.85, 0.18, 0.05])
    models: list[str] = []

    def _provider() -> RoutingContext:
        return RoutingContext(budget_remaining_ratio=next(ratios))

    evaluating.set_context_provider(_provider)
    evaluations: list[object] = []
    evaluating.set_on_evaluated(evaluations.append)

    for idx in range(3):
        evaluating.generate_messages(
            [ChatMessage(role="user", content=f"step-{idx}")],
            run_id="run-soak",
        )
        models.append(evaluating.model)

    assert len(evaluations) == 3

    assert models[0] == "gpt-4o-mini"
    assert models[1] == "meta-llama/Llama-3.1-8B"
    assert models[2] == "meta-llama/Llama-3.1-8B"

    report = state.llm_usage_tracker.build_report() if state.llm_usage_tracker else None
    assert report is not None
    assert report.total.calls == 3
