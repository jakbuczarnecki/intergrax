# © Artur Czarnecki. All rights reserved.

"""ACP-TOK-3 — budget reaction policies and runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.agents.acp_budget_reactions import maybe_emit_budget_threshold
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.acp_session_host import (
    ACP_HOST_CONTEXT_KEY,
    ACPSessionHostContext,
)
from intergrax.agents.authoring.budget_enforcing_llm_router import wrap_budget_enforcing_router
from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.contracts.agent_budget import (
    AgentBudgetSlice,
    BudgetExceededReaction,
    BudgetLimitEnforcement,
    BudgetNotifyChannel,
    BudgetReactionProfile,
    ResolvedBudgetLimits,
)
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.notifications.models import NotificationMessage
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
        await step_ctx.llm_router.complete(
            "one two three four five six seven eight nine ten",
            model_hint="premium",
        )
        if step_ctx.step_index >= 1:
            return StepOutcome.complete(
                output={"step": step_ctx.step_index},
                terminal_reason=TerminalReason.GOAL_MET,
            )
        return StepOutcome.continue_with(state_delta={"phase": "more"})


@dataclass
class _RecordingNotificationAdapter:
    messages: list[NotificationMessage] = field(default_factory=list)

    async def notify(self, message: NotificationMessage) -> None:
        self.messages.append(message)


@dataclass
class _RecordingBudgetHook:
    thresholds: list[dict] = field(default_factory=list)
    exceeded: list[dict] = field(default_factory=list)

    async def on_budget_threshold(self, payload: dict) -> None:
        self.thresholds.append(payload)

    async def on_budget_exceeded(self, payload: dict) -> None:
        self.exceeded.append(payload)


def _host_with_reaction(
    reaction: BudgetReactionProfile,
    *,
    token_limit: int = 8,
    notification_adapter: _RecordingNotificationAdapter | None = None,
    budget_hook: _RecordingBudgetHook | None = None,
) -> ACPSessionHostContext:
    environment = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="tok_reaction.lab"
    ).model_copy(
        update={
            "cost_profile": ApplicationEnvironmentProfile.lab_defaults()
            .cost_profile.model_copy(
                update={
                    "max_total_tokens": 10_000,
                    "budget_reaction": reaction,
                }
            )
        }
    )
    binding = AgentBinding.model_construct(
        contract_id="repeat-llm",
        capabilities=["demo.repeat_llm"],
        budget_slice=AgentBudgetSlice(
            max_total_tokens=token_limit,
            enforcement=BudgetLimitEnforcement.HARD,
        ),
    )
    return ACPSessionHostContext(
        app_profile=environment,
        binding=binding,
        notification_adapter=notification_adapter,
        budget_reaction_hook=budget_hook,
    )


@pytest.mark.unit
@pytest.mark.gate
async def test_budget_exceeded_hitl_pauses_run() -> None:
    host_ctx = _host_with_reaction(
        BudgetReactionProfile(on_agent_limit_exceeded=BudgetExceededReaction.HITL),
    )
    request = AgentRunRequest(
        input="hitl",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    result = await _RepeatLlmAgent().run(request)
    assert result.status == AgentRunStatus.PAUSED
    assert result.terminal_reason == TerminalReason.HUMAN_REQUIRED


@pytest.mark.unit
@pytest.mark.gate
async def test_budget_exceeded_notify_only_sends_notification() -> None:
    adapter = _RecordingNotificationAdapter()
    host_ctx = _host_with_reaction(
        BudgetReactionProfile(
            on_agent_limit_exceeded=BudgetExceededReaction.NOTIFY_ONLY,
            notify_channels=[BudgetNotifyChannel.SLACK],
            user_message_template="Budget cap reached for legal agent",
        ),
        notification_adapter=adapter,
    )
    request = AgentRunRequest(
        input="notify",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    result = await _RepeatLlmAgent().run(request)
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.BUDGET_EXCEEDED
    assert len(adapter.messages) == 1
    assert adapter.messages[0].channel == "slack"


@pytest.mark.unit
@pytest.mark.gate
async def test_budget_custom_hook_invoked_on_exceed() -> None:
    hook = _RecordingBudgetHook()
    host_ctx = _host_with_reaction(
        BudgetReactionProfile(on_agent_limit_exceeded=BudgetExceededReaction.ABORT),
        budget_hook=hook,
    )
    request = AgentRunRequest(
        input="hook",
        identity=RequestIdentity(tenant_id="tenant-1", user_id="user-1"),
        metadata={ACP_HOST_CONTEXT_KEY: host_ctx},
    )
    await _RepeatLlmAgent().run(request)
    assert len(hook.exceeded) == 1
    assert hook.exceeded[0]["scope"] == "agent"


@pytest.mark.unit
@pytest.mark.gate
async def test_degrade_model_forces_cheapest_allowed_model() -> None:
    router = StepLLMRouter(
        allowed_models=("premium", "economy"),
        default_model="premium",
    )
    kernel_ctx = StepKernelContext(agent_id="demo", budget_degrade_active=True)
    wrapped = wrap_budget_enforcing_router(
        router,
        limits=ResolvedBudgetLimits(
            agent_tokens_limit=1_000,
            agent_enforcement=BudgetLimitEnforcement.HARD,
        ),
        usage_provider=lambda: AcpInvocationUsageView(),
        degrade_provider=lambda: kernel_ctx.budget_degrade_active,
    )
    result = await wrapped.complete("hello world", model_hint="premium")
    assert result.model_id == "economy"


@pytest.mark.unit
@pytest.mark.gate
async def test_budget_threshold_emits_runtime_event() -> None:
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-1",
        task_id="task-1",
        budget_reaction=BudgetReactionProfile(
            warn_threshold_ratio=0.50,
            on_agent_limit_exceeded=BudgetExceededReaction.DEGRADE_MODEL,
        ),
        resolved_budget_limits=ResolvedBudgetLimits(
            agent_tokens_limit=20,
            limit_source="binding",
        ),
    )
    step_ctx = AgentStepContext(
        invocation_usage=AcpInvocationUsageView(
            agent=AcpTokenUsage(tokens_total=12, tokens_limit=20),
            environment=AcpTokenUsage(tokens_total=12, tokens_limit=20),
        ),
    )
    await maybe_emit_budget_threshold(step_ctx, kernel_ctx)
    assert kernel_ctx.budget_degrade_active is True
    assert any(
        event.event_type == RuntimeEventType.BUDGET_THRESHOLD for event in kernel_ctx.events
    )
