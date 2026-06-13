# © Artur Czarnecki. All rights reserved.

"""ACP-TOK-1 — token metering rollups in invocation state."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.agents.authoring.acp_session_host import (
    ACP_HOST_CONTEXT_KEY,
    ACPSessionHostContext,
)
from intergrax.agents.run_environment import merge_environment
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.acp_state import ACP_STATE_KEY, ACP_USAGE_KEY
from intergrax.contracts.agent_budget import AgentBudgetSlice, BudgetLimitEnforcement
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentExecutionOptions, AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


def _stub_build_context(_agent: IntergraxAgent, _request: RuntimeRequest) -> RuntimeContext:
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


class _LlmMeterAgent(IntergraxAgent):
    contract_id = "llm-meter"
    capabilities = ("demo.llm_meter",)
    agent_name = "LlmMeter"
    agent_description = "Meters LLM tokens"
    risk_level = AgentRiskLevel.LOW
    max_steps = 3
    models_used: list[str] = []

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _stub_build_context(self, request)

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        assert step_ctx.llm_router is not None
        model_hint = "frontier"
        usage = step_ctx.invocation_usage
        if (
            usage is not None
            and usage.environment.tokens_limit is not None
            and usage.environment.tokens_total
            >= int(usage.environment.tokens_limit * 0.5)
        ):
            model_hint = "economy"
        llm_result = await step_ctx.llm_router.complete("ping", model_hint=model_hint)
        _LlmMeterAgent.models_used.append(llm_result.model_id)
        if step_ctx.step_index >= 1:
            return StepOutcome.complete(
                output={"model": llm_result.model_id},
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(state_delta={"phase": "second"})


@pytest.mark.unit
@pytest.mark.gate
def test_merge_environment_materializes_token_limits() -> None:
    manifest = ApplicationManifest.lab(
        app_id="tok_merge_test",
        name="Tok Merge",
        route_prefix="/v1/tok_merge",
        env_prefix="TOK_MERGE_",
        agents=[],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="tok_merge.lab").model_copy(
        update={
            "cost_profile": ApplicationEnvironmentProfile.lab_defaults().cost_profile.model_copy(
                update={"max_total_tokens": 50_000}
            )
        }
    )
    binding = AgentBinding.model_construct(
        contract_id="llm-meter",
        capabilities=["demo.llm_meter"],
        budget_slice=AgentBudgetSlice(
            max_total_tokens=8_000,
            enforcement=BudgetLimitEnforcement.HARD,
        ),
    )
    request = AgentRunRequest(
        input="go",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        execution_options=AgentExecutionOptions(max_total_tokens=4_000),
    )
    merged = merge_environment(
        contract=_LlmMeterAgent().get_contract(),
        request=request,
        app_profile=environment,
        binding=binding,
    )
    assert merged.resolved_budget_limits.agent_tokens_limit == 4_000
    assert merged.resolved_budget_limits.agent_tokens_remaining == 4_000
    assert merged.resolved_budget_limits.environment_tokens_limit == 50_000
    assert merged.resolved_budget_limits.limit_source == "request"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
async def test_kernel_increments_budget_after_llm_drain() -> None:
    from intergrax.agents.authoring.llm_router import StepLLMRouter
    from intergrax.contracts.agent_budget import ResolvedBudgetLimits

    router = StepLLMRouter(
        allowed_models=("balanced", "economy", "frontier"),
        default_model="frontier",
    )
    await router.complete("one two three four", model_hint="balanced")
    step_ctx = AgentStepContext(
        step_index=0,
        metadata={},
        llm_router=router,
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-meter-1",
        state_root={
            ACP_STATE_KEY: {
                "schema_version": "acp.state.v1",
                "_version": 0,
                "budget": {
                    "tokens_limit": 10_000,
                    "tokens_remaining": 10_000,
                },
            }
        },
        resolved_budget_limits=ResolvedBudgetLimits(
            agent_tokens_limit=10_000,
            agent_tokens_remaining=10_000,
            environment_tokens_limit=20_000,
            environment_tokens_remaining=20_000,
            limit_source="binding",
        ),
    )
    outcome = StepOutcome.continue_with({"phase": "ok"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.outcome_applied is True
    assert record.step_record is not None
    assert record.step_record.llm_calls
    budget = kernel_ctx.state_root[ACP_STATE_KEY]["budget"]
    assert budget["tokens_total"] > 0
    assert budget["llm_calls"] == 1
    assert step_ctx.invocation_usage is not None
    assert step_ctx.invocation_usage.agent.tokens_total == budget["tokens_total"]
    assert step_ctx.invocation_usage.environment.tokens_total == budget["tokens_total"]
    assert ACP_USAGE_KEY in step_ctx.metadata


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
async def test_acp_run_persists_usage_metadata_and_budget_state() -> None:
    _LlmMeterAgent.models_used = []
    manifest = ApplicationManifest.lab(
        app_id="tok_run_test",
        name="Tok Run",
        route_prefix="/v1/tok_run",
        env_prefix="TOK_RUN_",
        agents=[],
    )
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="tok_run.lab").model_copy(
        update={
            "cost_profile": ApplicationEnvironmentProfile.lab_defaults().cost_profile.model_copy(
                update={"max_total_tokens": 100}
            )
        }
    )
    binding = AgentBinding.model_construct(
        contract_id="llm-meter",
        capabilities=["demo.llm_meter"],
        budget_slice=AgentBudgetSlice(max_total_tokens=100),
    )
    host_ctx = ACPSessionHostContext(
        app_profile=environment,
        binding=binding,
    )
    request = AgentRunRequest(
        input="meter",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    agent = _LlmMeterAgent()
    result = await agent.run(request)
    assert result.status.value == "succeeded"
    budget = result.state[ACP_STATE_KEY]["budget"]
    assert budget["tokens_total"] > 0
    assert budget["tokens_limit"] == 100
    assert ACP_USAGE_KEY in request.metadata
    env_usage = request.metadata[ACP_USAGE_KEY]["environment"]
    assert env_usage["tokens_total"] == budget["tokens_total"]


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
async def test_second_step_sees_invocation_usage_from_prior_llm_call() -> None:
    seen_totals: list[int] = []

    class _UsageObserverAgent(_LlmMeterAgent):
        async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
            if step_ctx.invocation_usage is not None:
                seen_totals.append(step_ctx.invocation_usage.agent.tokens_total)
            return await super().on_next_step(step_ctx)

    _UsageObserverAgent.models_used = []
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="tok_observer.lab")
    binding = AgentBinding.model_construct(
        contract_id="llm-meter",
        capabilities=["demo.llm_meter"],
    )
    host_ctx = ACPSessionHostContext(app_profile=environment, binding=binding)
    request = AgentRunRequest(
        input="observe",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    await _UsageObserverAgent().run(request)
    assert seen_totals[0] == 0
    assert seen_totals[1] > 0
