# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Notification adapters for long-running task progress (§18, F.4)."""

from __future__ import annotations

import logging
import os
from typing import Optional, Protocol, runtime_checkable

from intergrax.runtime.long_running.models import NotificationMessage

logger = logging.getLogger(__name__)

ENV_SLACK_WEBHOOK_URL = "INTERGRAX_SLACK_WEBHOOK_URL"
ENV_TEAMS_WEBHOOK_URL = "INTERGRAX_TEAMS_WEBHOOK_URL"


@runtime_checkable
class NotificationAdapter(Protocol):
    async def notify(self, message: NotificationMessage) -> None: ...


class LoggingNotificationAdapter:
    """Laboratory default — logs notification payloads without external I/O."""

    async def notify(self, message: NotificationMessage) -> None:
        logger.info(
            "notification channel=%s task=%s subject=%s body=%s",
            message.channel,
            message.task_id,
            message.subject,
            message.body,
        )


class SlackNotificationAdapter:
    """
    Slack webhook adapter (stub).

    Posts only when INTERGRAX_SLACK_WEBHOOK_URL is set; otherwise no-op.
    """

    def __init__(self, *, webhook_url: Optional[str] = None) -> None:
        self._webhook_url = webhook_url or os.environ.get(ENV_SLACK_WEBHOOK_URL, "").strip()

    async def notify(self, message: NotificationMessage) -> None:
        if not self._webhook_url:
            return
        # Laboratory stub: external delivery is opt-in via env; no network in default path.
        logger.info(
            "slack notification queued task=%s subject=%s",
            message.task_id,
            message.subject,
        )


class TeamsNotificationAdapter:
    """
    Microsoft Teams webhook adapter (stub).

    Posts only when INTERGRAX_TEAMS_WEBHOOK_URL is set; otherwise no-op.
    """

    def __init__(self, *, webhook_url: Optional[str] = None) -> None:
        self._webhook_url = webhook_url or os.environ.get(ENV_TEAMS_WEBHOOK_URL, "").strip()

    async def notify(self, message: NotificationMessage) -> None:
        if not self._webhook_url:
            return
        logger.info(
            "teams notification queued task=%s subject=%s",
            message.task_id,
            message.subject,
        )


def resolve_notification_adapter(channel: Optional[str]) -> NotificationAdapter:
    normalized = (channel or "log").strip().lower()
    if normalized == "slack":
        return SlackNotificationAdapter()
    if normalized == "teams":
        return TeamsNotificationAdapter()
    return LoggingNotificationAdapter()
