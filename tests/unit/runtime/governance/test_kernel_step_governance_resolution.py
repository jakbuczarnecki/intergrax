# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run_enums import AgentRunErrorCode
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.governance.kernel_step_governance_resolution import (
    governance_resolution_from_kernel_step_record,
)
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine


class _CountingPolicyEngine(RuntimePolicyEngine):
    def __init__(self) -> None:
        super().__init__()
        self.evaluate_decision_calls = 0

    def evaluate_decision(self, decision, *, context=None):
        self.evaluate_decision_calls += 1
        return super().evaluate_decision(decision, context=context)


@pytest.mark.unit
def test_kernel_policy_deny_preserves_canonical_policy_pre() -> None:
    policy_pre = PolicyDecision(
        action=PolicyAction.DENY,
        reason="custom_deny_reason",
        policy_rule_id="custom.rule.deny",
        enforcement_level=EnforcementLevel.MANDATORY,
        audit_payload={"proof": "r3"},
    )
    record = StepExecutionRecord(
        outcome_applied=False,
        policy_pre=policy_pre,
        error_code=AgentRunErrorCode.POLICY_DENIED,
    )
    agent_decision = AgentDecision(
        type=AgentDecisionType.FAIL,
        reason="policy_denied",
    )
    resolution = governance_resolution_from_kernel_step_record(record, agent_decision)
    assert resolution is not None
    assert resolution.policy_decision.action is PolicyAction.DENY
    assert resolution.policy_decision.reason == "custom_deny_reason"
    assert resolution.policy_decision.policy_rule_id == "custom.rule.deny"
    assert resolution.policy_decision.enforcement_level is EnforcementLevel.MANDATORY
    assert resolution.policy_decision.audit_payload == {"proof": "r3"}
    assert resolution.agent_decision.type is AgentDecisionType.FAIL
    assert resolution.should_block_execution is True


@pytest.mark.unit
def test_kernel_policy_deny_not_re_evaluated_as_agent_fail() -> None:
    policy_pre = PolicyDecision(
        action=PolicyAction.DENY,
        reason="test_deny_continue",
        policy_rule_id="test.deny_continue",
    )
    record = StepExecutionRecord(
        outcome_applied=False,
        policy_pre=policy_pre,
        error_code=AgentRunErrorCode.POLICY_DENIED,
    )
    agent_decision = AgentDecision(type=AgentDecisionType.FAIL, reason="policy_denied")
    counting = _CountingPolicyEngine()
    handler = ExecutionInterruptHandler(counting)
    resolution = governance_resolution_from_kernel_step_record(record, agent_decision)
    assert resolution is not None
    assert counting.evaluate_decision_calls == 0
    handler_resolution = handler.resolve_decision(
        agent_decision,
        task_id="t",
        run_id="r",
        agent_id="a",
        step_id="s",
    )
    assert handler_resolution.policy_decision.action is PolicyAction.ALLOW
    assert counting.evaluate_decision_calls == 1


@pytest.mark.unit
def test_genuine_agent_fail_still_uses_interrupt_handler() -> None:
    counting = _CountingPolicyEngine()
    handler = ExecutionInterruptHandler(counting)
    decision = AgentDecision(type=AgentDecisionType.FAIL, reason="agent_failed")
    resolution = handler.resolve_decision(
        decision,
        task_id="t",
        run_id="r",
        agent_id="a",
        step_id="s",
    )
    assert counting.evaluate_decision_calls == 1
    assert resolution.agent_decision.type is AgentDecisionType.FAIL


@pytest.mark.unit
def test_genuine_agent_fail_with_allow_policy() -> None:
    handler = ExecutionInterruptHandler(PolicyEngine())
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.FAIL, reason="agent_failed"),
        task_id="t",
        run_id="r",
        agent_id="a",
        step_id="s",
    )
    assert resolution.policy_decision.action is PolicyAction.ALLOW
    assert resolution.agent_decision.type is AgentDecisionType.FAIL
    assert resolution.should_fail is True
    assert resolution.should_block_execution is False


@pytest.mark.unit
def test_kernel_record_without_policy_pre_falls_through() -> None:
    record = StepExecutionRecord(
        outcome_applied=False,
        error_code=AgentRunErrorCode.POLICY_DENIED,
    )
    resolution = governance_resolution_from_kernel_step_record(
        record,
        AgentDecision(type=AgentDecisionType.FAIL, reason="policy_denied"),
    )
    assert resolution is None
