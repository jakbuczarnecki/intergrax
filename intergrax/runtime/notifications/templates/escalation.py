# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Escalation notification templates (§42.38, B.05)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intergrax.runtime.human.models import EscalationOutcome
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.task.task import Task

ESCALATION_TEMPLATE_ID = "escalation.v1"


@dataclass(frozen=True)
class EscalationNotificationContext:
    level: int
    target: str
    message: str
    progress_message: str
    resume_token: Optional[str]
    checkpoint_id: Optional[str]
    escalation_chain: list[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        outcome: EscalationOutcome,
        progress_message: str,
    ) -> EscalationNotificationContext:
        gov = task.runtime.governance
        chain = [
            {"level": step.level, "target": step.target, "message": step.message}
            for step in gov.escalation_chain
        ]
        return cls(
            level=outcome.level,
            target=outcome.target.value,
            message=outcome.message,
            progress_message=progress_message,
            resume_token=task.runtime.orchestration.resume_token,
            checkpoint_id=task.runtime.orchestration.checkpoint_id,
            escalation_chain=chain,
        )


class EscalationNotificationTemplate:
    SUBJECT = "Task escalation required"

    @staticmethod
    def render(context: EscalationNotificationContext) -> tuple[str, str, Dict[str, Any]]:
        body_parts = [
            context.progress_message,
            "",
            f"Escalation level: {context.level}",
            f"Target: {context.target}",
        ]
        if context.message:
            body_parts.extend(["", context.message])
        if context.resume_token:
            body_parts.extend(["", f"Resume token: `{context.resume_token}`"])
        if context.escalation_chain:
            body_parts.extend(["", "Escalation chain:"])
            for step in context.escalation_chain:
                body_parts.append(
                    f"- level {step.get('level')}: {step.get('target')} — {step.get('message', '')}"
                )
        metadata: Dict[str, Any] = {
            "template": ESCALATION_TEMPLATE_ID,
            "escalation_level": context.level,
            "escalation_target": context.target,
            "escalation_chain": context.escalation_chain,
        }
        return EscalationNotificationTemplate.SUBJECT, "\n".join(body_parts), metadata


def build_escalation_notification_message(
    task: Task,
    *,
    outcome: EscalationOutcome,
    progress_message: str,
    channel: str,
) -> NotificationMessage:
    context = EscalationNotificationContext.from_task(
        task,
        outcome=outcome,
        progress_message=progress_message,
    )
    subject, body, metadata = EscalationNotificationTemplate.render(context)
    return NotificationMessage(
        channel=channel,
        subject=subject,
        body=body,
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        metadata={
            "task_state": task.state.value,
            "resume_token": context.resume_token,
            "checkpoint_id": context.checkpoint_id,
            **metadata,
        },
    )


def is_escalation_templated_message(message: NotificationMessage) -> bool:
    return message.metadata.get("template") == ESCALATION_TEMPLATE_ID
