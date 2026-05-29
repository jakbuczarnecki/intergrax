# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Partial-result notification templates for long-running tasks (§26, J.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.task.task import Task

PARTIAL_RESULT_TEMPLATE_ID = "partial_result.v1"


@dataclass(frozen=True)
class PartialResultNotificationContext:
    progress_message: str
    task_state: str
    resume_token: Optional[str]
    checkpoint_id: Optional[str]
    last_step_summary: Optional[str] = None
    partial_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        progress_message: str,
        partial_payload: Optional[Dict[str, Any]] = None,
        last_step_summary: Optional[str] = None,
    ) -> PartialResultNotificationContext:
        return cls(
            progress_message=progress_message,
            task_state=task.state.value,
            resume_token=task.runtime.orchestration.resume_token,
            checkpoint_id=task.runtime.orchestration.checkpoint_id,
            last_step_summary=last_step_summary,
            partial_payload=dict(partial_payload or {}),
        )


class PartialResultNotificationTemplate:
    """Format partial progress for log/Slack/Teams notification stubs."""

    template_id = PARTIAL_RESULT_TEMPLATE_ID

    def render(self, context: PartialResultNotificationContext) -> PartialResultNotificationContent:
        body_parts = [context.progress_message or "Task progress update"]
        if context.last_step_summary:
            body_parts.extend(["", f"Last step: {context.last_step_summary}"])
        if context.resume_token:
            body_parts.extend(["", f"Resume token: {context.resume_token}"])
        body = "\n".join(body_parts)
        metadata = {
            "template": self.template_id,
            "task_state": context.task_state,
            "checkpoint_id": context.checkpoint_id,
            "resume_token": context.resume_token,
            "last_step_summary": context.last_step_summary,
            "partial_payload": context.partial_payload,
        }
        return PartialResultNotificationContent(
            subject=f"Task progress ({context.task_state})",
            body=body,
            metadata=metadata,
        )


@dataclass(frozen=True)
class PartialResultNotificationContent:
    subject: str
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_partial_result_notification_message(
    task: Task,
    *,
    progress_message: str,
    channel: str,
    partial_payload: Optional[Dict[str, Any]] = None,
    last_step_summary: Optional[str] = None,
) -> NotificationMessage:
    context = PartialResultNotificationContext.from_task(
        task,
        progress_message=progress_message,
        partial_payload=partial_payload,
        last_step_summary=last_step_summary,
    )
    content = PartialResultNotificationTemplate().render(context)
    return NotificationMessage(
        channel=channel,
        subject=content.subject,
        body=content.body,
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        metadata=content.metadata,
    )


def is_partial_result_templated_message(message: NotificationMessage) -> bool:
    return message.metadata.get("template") == PARTIAL_RESULT_TEMPLATE_ID
