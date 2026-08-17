# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.agents.authoring.llm_router import StepLLMRouter
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPhase, PreModelPolicyContext
from intergrax.runtime.governance.post_run_governance_bridge import invoke_post_run_governance
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.policy.agent_decision_enforcement import (
    agent_decision_failure_from_resolution,
)
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_bridge import (
    PreModelPolicyBlockedError,
    wrap_policy_enforcing_llm_router,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


class _DenyContinuePolicyEngine(RuntimePolicyEngine):
    def evaluate_decision(self, decision, *, context=None):
        if decision.type is AgentDecisionType.CONTINUE:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="test_deny_continue",
                policy_rule_id="test.deny_continue",
            )
        return super().evaluate_decision(decision, context=context)


class _RecordingGovernanceService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def evaluate(self, run_id: str, agent_id: str):
        self.calls.append((run_id, agent_id))
        return None


@pytest.mark.unit
def test_governance_resolution_should_block_execution_on_deny() -> None:
    resolution = GovernanceResolution(
        policy_decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        ),
        agent_decision=AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
    )
    assert resolution.should_block_execution is True
    assert resolution.should_pause is False


@pytest.mark.unit
def test_agent_decision_enforcement_maps_deny_to_fail() -> None:
    resolution = GovernanceResolution(
        policy_decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        ),
        agent_decision=AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
    )
    failed = agent_decision_failure_from_resolution(resolution)
    assert failed.type is AgentDecisionType.FAIL


@pytest.mark.unit
def test_interrupt_handler_deny_blocks_execution() -> None:
    handler = ExecutionInterruptHandler(_DenyContinuePolicyEngine())
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id="task_1",
        run_id="run_1",
        agent_id="agent_1",
        step_id="step_1",
    )
    assert resolution.should_block_execution is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_model_policy_blocks_provider_before_complete() -> None:
    called = {"value": False}

    class _Port:
        async def complete(self, prompt: str, *, model_id: str, provider: str):
            called["value"] = True
            return "ok", 1, 1

    inner = StepLLMRouter(
        allowed_models=("balanced",),
        default_model="balanced",
        llm_port=_Port(),
    )
    router = wrap_policy_enforcing_llm_router(
        inner,
        policy_engine=PolicyEngine(),
        tenant_id="tenant_1",
        agent_id="agent_1",
    )

    class _DenyAgentModelEngine(RuntimePolicyEngine):
        def evaluate_pre_llm(self, *, tenant_id, agent_id, message_count, context=None):
            ctx = context or PreModelPolicyContext()
            if ctx.phase is PreModelPhase.AGENT_STEP and ctx.model_id == "balanced":
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="agent_model_denied",
                    policy_rule_id="test.agent_model_denied",
                )
            return super().evaluate_pre_llm(
                tenant_id=tenant_id,
                agent_id=agent_id,
                message_count=message_count,
                context=context,
            )

    router._policy_engine = PolicyEngine(runtime=_DenyAgentModelEngine())  # noqa: SLF001

    with pytest.raises(PreModelPolicyBlockedError):
        await router.complete("hello")

    assert called["value"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_model_policy_allows_provider_on_allow() -> None:
    called = {"value": False}

    class _Port:
        async def complete(self, prompt: str, *, model_id: str, provider: str):
            called["value"] = True
            return "ok", 1, 1

    inner = StepLLMRouter(
        allowed_models=("balanced",),
        default_model="balanced",
        llm_port=_Port(),
    )
    router = wrap_policy_enforcing_llm_router(
        inner,
        policy_engine=PolicyEngine(),
        tenant_id="tenant_1",
        agent_id="agent_1",
    )
    result = await router.complete("hello")
    assert called["value"] is True
    assert result.text == "ok"


@pytest.mark.unit
def test_post_run_governance_bridge_invokes_service() -> None:
    service = _RecordingGovernanceService()
    invoke_post_run_governance(service, run_id="run_1", agent_id="agent_1")
    assert service.calls == [("run_1", "agent_1")]
