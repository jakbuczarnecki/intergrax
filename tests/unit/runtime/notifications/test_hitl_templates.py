# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_decision import AgentDecisionType, HumanRequest, HumanRequestUrgency
from intergrax.runtime.notifications.formatters import SlackPayloadFormatter, TeamsPayloadFormatter
from intergrax.runtime.notifications.templates.hitl import (
    HITL_PAUSE_TEMPLATE_ID,
    HitlPauseNotificationContext,
    HitlPauseNotificationTemplate,
    build_hitl_actions,
    build_hitl_pause_notification_message,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import (
    TaskExecutionOptions,
    TaskGovernanceState,
    TaskLongRunningOptions,
    TaskRuntimeState,
)


def _task_with_human_request(**overrides) -> Task:
    human_request = HumanRequest(
        request_id="hr_template_1",
        prompt="Approve vendor onboarding?",
        options=["approve", "reject", "escalate"],
        urgency=HumanRequestUrgency.HIGH,
        timeout_seconds=900,
        default_on_timeout=AgentDecisionType.ESCALATE,
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="vendor onboarding",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel="slack"),
        ),
        runtime=TaskRuntimeState(
            governance=TaskGovernanceState(
                human_request=human_request,
                human_request_created_at="2026-05-27T10:00:00+00:00",
                human_request_expires_at="2026-05-27T10:15:00+00:00",
            ),
            orchestration={"checkpoint_id": "ckpt_1", "resume_token": "tok_abc"},
        ),
    )
    task.runtime.orchestration.checkpoint_id = "ckpt_1"
    task.runtime.orchestration.resume_token = "tok_abc"
    task.state = TaskState.WAITING_FOR_HUMAN
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


@pytest.mark.unit
@pytest.mark.gate
def test_build_hitl_actions_deduplicates_synonyms():
    actions = build_hitl_actions(["approve", "yes", "reject"])
    assert [action.action_id for action in actions] == ["approve", "reject"]


@pytest.mark.unit
@pytest.mark.gate
def test_hitl_pause_template_renders_resume_token_and_actions():
    task = _task_with_human_request()
    content = HitlPauseNotificationTemplate.render(
        HitlPauseNotificationContext.from_task(task, progress_message="awaiting human input")
    )
    assert content.metadata["template"] == HITL_PAUSE_TEMPLATE_ID
    assert "tok_abc" in content.body
    assert "Approve vendor onboarding?" in content.body
    assert "reply with `approve`" in content.body
    assert "reply with `reject`" in content.body
    assert content.metadata["urgency"] == "high"
    assert content.metadata["expires_at_utc"] == "2026-05-27T10:15:00+00:00"
    assert len(content.metadata["actions"]) == 3


@pytest.mark.unit
@pytest.mark.gate
def test_build_hitl_pause_notification_message():
    task = _task_with_human_request()
    message = build_hitl_pause_notification_message(
        task,
        progress_message="awaiting human input",
        channel="slack",
    )
    assert message.channel == "slack"
    assert message.metadata["resume_token"] == "tok_abc"
    assert message.metadata["human_request_id"] == "hr_template_1"


@pytest.mark.unit
@pytest.mark.gate
def test_slack_formatter_keeps_hitl_template_body():
    task = _task_with_human_request()
    message = build_hitl_pause_notification_message(
        task,
        progress_message="awaiting human input",
        channel="slack",
    )
    payload = SlackPayloadFormatter().format(message)
    assert "Human approval required" in payload["text"]
    assert "reply with `approve`" in payload["text"]
    assert payload["text"].count("tok_abc") >= 1


@pytest.mark.unit
@pytest.mark.gate
def test_teams_formatter_includes_hitl_action_facts():
    task = _task_with_human_request()
    message = build_hitl_pause_notification_message(
        task,
        progress_message="awaiting human input",
        channel="teams",
    )
    payload = TeamsPayloadFormatter().format(message)
    facts = payload["sections"][0]["facts"]
    names = {item["name"] for item in facts}
    assert "Approve" in names
    assert "Reject" in names
    assert "Urgency" in names
