# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.human.models import EscalationOutcome, EscalationTarget
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.notifications.templates.escalation import (
    ESCALATION_TEMPLATE_ID,
    build_escalation_notification_message,
    is_escalation_templated_message,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions


@pytest.mark.unit
@pytest.mark.gate
def test_interrupt_handler_safety_violation_timeout_escalates() -> None:
    handler = ExecutionInterruptHandler()
    interrupt = ExecutionInterrupt(
        interrupt_type=InterruptType.SAFETY_VIOLATION,
        source_agent_id="agent_1",
        task_id="task_1",
        run_id="run_1",
        blocking=True,
    )

    resolution = handler.resolve_interrupt(interrupt)

    assert resolution.policy_decision.action == PolicyAction.REQUIRE_HUMAN
    assert resolution.human_request is not None
    assert resolution.human_request.urgency.value == "critical"
    assert resolution.human_request.timeout_seconds == 1800
    assert resolution.human_request.default_on_timeout is not None
    assert resolution.human_request.default_on_timeout.value == "escalate"
    assert "escalate" in resolution.human_request.options


@pytest.mark.unit
@pytest.mark.gate
def test_escalation_notification_template_metadata() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="review contract",
        context=TaskContext(capability="legal.contract_review"),
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="log"),
        ),
    )
    task.runtime.orchestration.resume_token = "rtok_1"
    task.runtime.orchestration.checkpoint_id = "ckpt_1"
    outcome = EscalationOutcome(
        target=EscalationTarget.HUMAN_OPERATOR,
        level=1,
        message="escalated to human operator",
    )

    message = build_escalation_notification_message(
        task,
        outcome=outcome,
        progress_message="awaiting escalated human review",
        channel="log",
    )

    assert isinstance(message, NotificationMessage)
    assert message.metadata["template"] == ESCALATION_TEMPLATE_ID
    assert message.metadata["escalation_level"] == 1
    assert message.metadata["escalation_target"] == "human_operator"
    assert is_escalation_templated_message(message)
    assert "Escalation level: 1" in message.body
