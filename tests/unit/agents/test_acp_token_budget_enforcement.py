# © Artur Czarnecki. All rights reserved.

"""ACP-TOK-2 — hard/advisory token budget enforcement."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.acp_session_host import (
    ACP_HOST_CONTEXT_KEY,
    ACPSessionHostContext,
)
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.acp_budget_enforcement import evaluate_hard_budget_violation
from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.contracts.agent_budget import AgentBudgetSlice, BudgetLimitEnforcement, ResolvedBudgetLimits
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    AgentRunStatus,
    TerminalReason,
)
from intergrax.contracts.agent_step_context import AgentStepContext
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


class _RepeatLlmAgent(IntergraxAgent):
    contract_id = "repeat-llm"
    capabilities = ("demo.repeat_llm",)
    agent_name = "RepeatLlm"
    agent_description = "Calls LLM every step"
    risk_level = AgentRiskLevel.LOW
    max_steps = 4

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return _stub_build_context(self, request)

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        assert step_ctx.llm_router is not None
        await step_ctx.llm_router.complete("one two three four five six seven eight")
        if step_ctx.step_index >= 1:
            return StepOutcome.complete(
                output={"step": step_ctx.step_index},
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(state_delta={"phase": "more"})


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_hard_budget_violation_blocks_at_limit() -> None:
    usage = AcpInvocationUsageView(
        agent=AcpTokenUsage(tokens_total=100, tokens_limit=100),
        environment=AcpTokenUsage(tokens_total=50, tokens_limit=200),
    )
    limits = ResolvedBudgetLimits(
        agent_tokens_limit=100,
        agent_enforcement=BudgetLimitEnforcement.HARD,
        environment_tokens_limit=200,
        environment_enforcement=BudgetLimitEnforcement.HARD,
        limit_source="binding",
    )
    violation = evaluate_hard_budget_violation(usage, limits)
    assert violation is not None
    assert violation.scope.value == "agent"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_hard_budget_advisory_does_not_block() -> None:
    usage = AcpInvocationUsageView(
        agent=AcpTokenUsage(tokens_total=500, tokens_limit=100),
        environment=AcpTokenUsage(tokens_total=500, tokens_limit=200),
    )
    limits = ResolvedBudgetLimits(
        agent_tokens_limit=100,
        agent_enforcement=BudgetLimitEnforcement.ADVISORY,
        environment_tokens_limit=200,
        environment_enforcement=BudgetLimitEnforcement.ADVISORY,
        limit_source="binding",
    )
    assert evaluate_hard_budget_violation(usage, limits) is None


@pytest.mark.unit
@pytest.mark.gate
async def test_hard_limit_blocks_second_step_llm() -> None:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="tok_hard.lab")
    binding = AgentBinding.model_construct(
        contract_id="repeat-llm",
        capabilities=["demo.repeat_llm"],
        budget_slice=AgentBudgetSlice(
            max_total_tokens=8,
            enforcement=BudgetLimitEnforcement.HARD,
        ),
    )
    host_ctx = ACPSessionHostContext(app_profile=environment, binding=binding)
    request = AgentRunRequest(
        input="enforce",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    result = await _RepeatLlmAgent().run(request)
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.BUDGET_EXCEEDED
    assert any(error.code == AgentRunErrorCode.BUDGET_EXCEEDED for error in result.errors)


@pytest.mark.unit
@pytest.mark.gate
async def test_advisory_limit_allows_second_step_llm() -> None:
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="tok_advisory.lab")
    binding = AgentBinding.model_construct(
        contract_id="repeat-llm",
        capabilities=["demo.repeat_llm"],
        budget_slice=AgentBudgetSlice(
            max_total_tokens=8,
            enforcement=BudgetLimitEnforcement.ADVISORY,
        ),
    )
    host_ctx = ACPSessionHostContext(app_profile=environment, binding=binding)
    request = AgentRunRequest(
        input="advisory",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    result = await _RepeatLlmAgent().run(request)
    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.terminal_reason == TerminalReason.GOAL_MET
