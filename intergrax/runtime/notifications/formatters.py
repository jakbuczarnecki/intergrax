# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Channel-agnostic payload formatters for outbound notifications."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from intergrax.runtime.notifications.models import NotificationMessage


@runtime_checkable
class NotificationPayloadFormatter(Protocol):
    """Maps a canonical ``NotificationMessage`` to a transport-specific body."""

    def format(self, message: NotificationMessage) -> Dict[str, Any]: ...


class GenericJsonPayloadFormatter:
    """Vendor-neutral JSON envelope — default for generic webhook backends."""

    def format(self, message: NotificationMessage) -> Dict[str, Any]:
        return {
            "channel": message.channel,
            "subject": message.subject,
            "body": message.body,
            "task_id": message.task_id,
            "tenant_id": message.tenant_id,
            "metadata": dict(message.metadata),
        }


class SlackPayloadFormatter:
    """Slack Incoming Webhook text payload."""

    def format(self, message: NotificationMessage) -> Dict[str, Any]:
        text = f"*{message.subject}*\n{message.body}"
        resume_token = message.metadata.get("resume_token")
        if resume_token:
            text += f"\n_resume token:_ `{resume_token}`"
        return {"text": text}


class TeamsPayloadFormatter:
    """Microsoft Teams Incoming Webhook MessageCard (simplified)."""

    def format(self, message: NotificationMessage) -> Dict[str, Any]:
        facts = [
            {"name": "Task", "value": message.task_id},
            {"name": "Tenant", "value": message.tenant_id},
        ]
        resume_token = message.metadata.get("resume_token")
        if resume_token:
            facts.append({"name": "Resume token", "value": str(resume_token)})
        return {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": message.subject,
            "themeColor": "0078D4",
            "title": message.subject,
            "text": message.body,
            "sections": [{"facts": facts}],
        }
