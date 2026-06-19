# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY, ACPSessionHostContext
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetExceededDegradeRule, LLMRoutingProfile, RoutingContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _RoutingTraceAgent(IntergraxAgent):
    contract_id = "routing-trace-agent"
    capabilities = ("demo.routing_trace",)
    agent_name = "RoutingTrace"
    agent_description = "ACP routing trace"
    risk_level = AgentRiskLevel.LOW
    max_steps = 1

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        assert step_ctx.llm_router is not None
        llm_result = await step_ctx.llm_router.complete("ping", model_hint="premium")
        return StepOutcome.complete(
            output={"answer": llm_result.text},
            terminal_reason=TerminalReason.GOAL_MET,
        )


@pytest.mark.unit
@pytest.mark.gate
async def test_acp_run_records_routing_rule_in_step_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        lambda _env, **kwargs: FakeLLMAdapter(),
    )

    def _simple_provider(**_kwargs: object):
        def _provider() -> RoutingContext:
            return RoutingContext(budget_degrade_active=True)

        return _provider

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_routing_context_bridge.make_acp_routing_context_provider",
        _simple_provider,
    )
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="premium")
    economy = LLMProfile(provider=LLMProvider.OPENAI, model="economy")
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="routing_trace.lab")
    environment = environment.model_copy(
        update={
            "llm_profile": primary,
            "llm_routing_profile": LLMRoutingProfile(
                default_profile=primary,
                allowed_profiles=(primary, economy),
                rules=(BudgetExceededDegradeRule(),),
            ),
        }
    )
    binding = AgentBinding.model_construct(
        contract_id="routing-trace-agent",
        capabilities=["demo.routing_trace"],
    )
    host_ctx = ACPSessionHostContext(app_profile=environment, binding=binding)
    request = AgentRunRequest(
        input="route",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={
            ACP_HOST_CONTEXT_KEY: host_ctx,
            "budget_degrade_active": True,
        },
    )
    result = await _RoutingTraceAgent().run(request)
    assert result.trace.steps
    diagnostics = result.trace.steps[0].diagnostics
    assert "llm_routing_evaluations" in diagnostics
    routing_diag = diagnostics["llm_routing_evaluations"][0]
    assert routing_diag["matched_rule_id"] == "builtin.budget_degrade"
    assert "budget_degrade_active" in routing_diag["routing_reason"]
    assert len(result.trace.steps[0].llm_calls) == 1
