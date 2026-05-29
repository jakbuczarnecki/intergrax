# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Notification adapters for long-running task progress (§18, F.4).

Slack wiring delegates to ``integrations.providers.slack``; Teams remains here until M.4 teams.
"""

from __future__ import annotations

import os
from typing import Optional

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.factory import (
    ENV_SLACK_WEBHOOK_URL,
    ENV_TEAMS_WEBHOOK_URL,
    resolve_notification_adapter as _resolve_notification_adapter,
)
from intergrax.runtime.notifications.formatters import TeamsPayloadFormatter

__all__ = [
    "ENV_SLACK_WEBHOOK_URL",
    "ENV_TEAMS_WEBHOOK_URL",
    "LoggingNotificationAdapter",
    "NotificationAdapter",
    "TeamsNotificationAdapter",
    "create_slack_notification_channel",
    "resolve_notification_adapter",
]

def create_slack_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Re-export — composition root is ``integrations.providers.slack``."""
    from intergrax.integrations.providers.slack.bundle import (
        create_slack_notification_channel as _create,
    )

    return _create(webhook_url=webhook_url, delivery=delivery, **config_overrides)


class TeamsNotificationAdapter(WebhookNotificationAdapter):
    """Microsoft Teams Incoming Webhook — composes ``TeamsPayloadFormatter`` + HTTP delivery."""

    def __init__(
        self,
        *,
        webhook_url: Optional[str] = None,
        delivery: Optional[NotificationDelivery] = None,
    ) -> None:
        super().__init__(
            webhook_url=(webhook_url or os.environ.get(ENV_TEAMS_WEBHOOK_URL, "")).strip(),
            formatter=TeamsPayloadFormatter(),
            delivery=delivery,
            channel="teams",
        )


def resolve_notification_adapter(
    channel: Optional[str],
    *,
    delivery: Optional[NotificationDelivery] = None,
) -> NotificationAdapter:
    normalized = (channel or "log").strip().lower()
    if normalized == "slack":
        return create_slack_notification_channel(delivery=delivery)
    if normalized == "teams":
        adapter = TeamsNotificationAdapter(delivery=delivery)
        return adapter if adapter.webhook_url else LoggingNotificationAdapter()
    return _resolve_notification_adapter(channel, delivery=delivery)
