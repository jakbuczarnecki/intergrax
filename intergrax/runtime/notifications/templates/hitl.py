# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable HITL pause notification templates (Phase H.4, §42.10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from intergrax.runtime.notifications.models import NotificationMessage

if TYPE_CHECKING:
    from intergrax.runtime.task.task import Task

HITL_PAUSE_TEMPLATE_ID = "hitl.pause.v1"

_OPTION_SYNONYMS: Dict[str, str] = {
    "approve": "approve",
    "yes": "approve",
    "y": "approve",
    "ok": "approve",
    "accept": "approve",
    "reject": "reject",
    "no": "reject",
    "n": "reject",
    "deny": "reject",
    "decline": "reject",
    "escalate": "escalate",
    "delegate": "delegate",
    "modify": "modify",
    "edit": "modify",
}

_DEFAULT_ACTION_LABELS: Dict[str, str] = {
    "approve": "Approve",
    "reject": "Reject",
    "escalate": "Escalate",
    "delegate": "Delegate",
    "modify": "Modify",
}


@dataclass(frozen=True)
class HitlNotificationAction:
    action_id: str
    label: str
    response_value: str

    def to_metadata(self) -> Dict[str, str]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "response_value": self.response_value,
        }


@dataclass(frozen=True)
class HitlNotificationContent:
    subject: str
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HitlPauseNotificationContext:
    request_id: str
    prompt: str
    options: List[str]
    urgency: str
    timeout_seconds: Optional[int]
    expires_at_utc: Optional[str]
    resume_token: Optional[str]
    checkpoint_id: Optional[str]
    progress_message: str

    @classmethod
    def from_task(cls, task: Task, *, progress_message: str) -> HitlPauseNotificationContext:
        gov = task.runtime.governance
        human_request = gov.human_request
        if human_request is None:
            raise ValueError("task has no active human_request")
        return cls(
            request_id=human_request.request_id,
            prompt=human_request.prompt,
            options=list(human_request.options),
            urgency=human_request.urgency.value,
            timeout_seconds=human_request.timeout_seconds,
            expires_at_utc=gov.human_request_expires_at,
            resume_token=task.runtime.orchestration.resume_token,
            checkpoint_id=task.runtime.orchestration.checkpoint_id,
            progress_message=progress_message,
        )


def _canonical_option(option: str) -> str:
    return _OPTION_SYNONYMS.get(option.strip().lower(), option.strip().lower())


def build_hitl_actions(options: List[str]) -> List[HitlNotificationAction]:
    seen: set[str] = set()
    actions: List[HitlNotificationAction] = []
    for raw in options:
        action_id = _canonical_option(raw)
        if not action_id or action_id in seen:
            continue
        seen.add(action_id)
        label = _DEFAULT_ACTION_LABELS.get(action_id, action_id.replace("_", " ").title())
        actions.append(
            HitlNotificationAction(
                action_id=action_id,
                label=label,
                response_value=action_id,
            )
        )
    return actions


def format_hitl_actions_text(actions: List[HitlNotificationAction]) -> str:
    lines = [f"- {action.label}: reply with `{action.response_value}`" for action in actions]
    return "\n".join(lines)


class HitlPauseNotificationTemplate:
    SUBJECT = "Human approval required"

    @staticmethod
    def render(context: HitlPauseNotificationContext) -> HitlNotificationContent:
        options = context.options or ["approve", "reject"]
        actions = build_hitl_actions(options)
        body_parts = [context.progress_message, "", context.prompt, ""]
        if context.resume_token:
            body_parts.extend([f"Resume token: `{context.resume_token}`", ""])
        body_parts.append(format_hitl_actions_text(actions))
        if context.expires_at_utc:
            body_parts.extend(["", f"Expires (UTC): {context.expires_at_utc}"])
        metadata: Dict[str, Any] = {
            "template": HITL_PAUSE_TEMPLATE_ID,
            "human_request_id": context.request_id,
            "urgency": context.urgency,
            "timeout_seconds": context.timeout_seconds,
            "expires_at_utc": context.expires_at_utc,
            "actions": [action.to_metadata() for action in actions],
        }
        return HitlNotificationContent(
            subject=HitlPauseNotificationTemplate.SUBJECT,
            body="\n".join(body_parts),
            metadata=metadata,
        )


def build_hitl_pause_notification_message(
    task: Task,
    *,
    progress_message: str,
    channel: str,
) -> NotificationMessage:
    context = HitlPauseNotificationContext.from_task(task, progress_message=progress_message)
    content = HitlPauseNotificationTemplate.render(context)
    return NotificationMessage(
        channel=channel,
        subject=content.subject,
        body=content.body,
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        metadata={
            "task_state": task.state.value,
            "resume_token": context.resume_token,
            "checkpoint_id": context.checkpoint_id,
            **content.metadata,
        },
    )


def is_hitl_templated_message(message: NotificationMessage) -> bool:
    return message.metadata.get("template") == HITL_PAUSE_TEMPLATE_ID
