# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty notification channel adapter."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from intergrax.integrations.providers.notification_channel.pagerduty.client import PagerDutyEventsClient
from intergrax.runtime.notifications.models import NotificationMessage


class PagerDutyNotificationChannel:
    """``NotificationChannel`` facade with explicit incident trigger."""

    def __init__(self, client: PagerDutyEventsClient) -> None:
        self._client = client

    @property
    def events_client(self) -> PagerDutyEventsClient:
        return self._client

    async def notify(self, message: NotificationMessage) -> None:
        self._client.send_notification(
            subject=str(message.subject or message.task_id),
            body=str(message.body or ""),
            task_id=str(message.task_id or "intergrax"),
        )

    def trigger_incident(
        self,
        *,
        summary: str,
        severity: str = "error",
        source: str = "intergrax",
        custom_details: Optional[Mapping[str, Any]] = None,
        dedup_key: Optional[str] = None,
    ) -> str:
        return self._client.trigger_incident(
            summary=summary,
            severity=severity,
            source=source,
            custom_details=custom_details,
            dedup_key=dedup_key,
        )
