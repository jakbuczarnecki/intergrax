# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_interrupt import InterruptType
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.policy.policy_engine import PolicyEngine


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_handler_maps_request_human_to_pause():
    handler = ExecutionInterruptHandler()
    decision = AgentDecision(
        type=AgentDecisionType.REQUEST_HUMAN,
        reason="needs review",
        human_request=HumanRequest(
            request_id="hr_test",
            prompt="Approve?",
            options=["approve", "reject"],
        ),
    )

    resolution = handler.resolve_decision(
        decision,
        task_id="task_1",
        run_id="run_1",
        agent_id="agent_1",
        step_id="step_1",
    )

    assert resolution.should_pause is True
    assert resolution.human_request is not None
    assert resolution.human_request.request_id == "hr_test"


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_handler_critical_interrupt_requires_human():
    handler = ExecutionInterruptHandler(PolicyEngine())
    decision = AgentDecision(
        type=AgentDecisionType.INTERRUPT,
        reason="critical safety issue",
        severity=EventSeverity.CRITICAL,
        payload={"interrupt_type": InterruptType.SAFETY_VIOLATION.value},
    )

    resolution = handler.resolve_decision(
        decision,
        task_id="task_2",
        run_id="run_2",
        agent_id="agent_2",
    )

    assert resolution.interrupt is not None
    assert resolution.policy_decision.action == PolicyAction.REQUIRE_HUMAN
    assert resolution.should_pause is True


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_handler_modify_plan_without_handoff_fails():
    handler = ExecutionInterruptHandler()
    decision = AgentDecision(
        type=AgentDecisionType.MODIFY_PLAN,
        reason="replan requested",
    )

    resolution = handler.resolve_decision(
        decision,
        task_id="task_3",
        run_id="run_3",
        agent_id="agent_3",
        step_id="step_3",
    )

    assert resolution.should_fail is True
    assert resolution.policy_decision.action == PolicyAction.DENY
    assert resolution.policy_decision.reason == "MODIFY_PLAN_NOT_SUPPORTED"
