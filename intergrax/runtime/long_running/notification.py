# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Notification adapters for long-running task progress (§18, F.4).

Slack, Teams, and generic webhook wiring delegate to ``integrations.providers.*``.
"""

from __future__ import annotations

from typing import Optional

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.factory import (
    ENV_SLACK_WEBHOOK_URL,
    ENV_TEAMS_WEBHOOK_URL,
    ENV_WEBHOOK_URL,
    resolve_notification_adapter as _resolve_notification_adapter,
)

__all__ = [
    "ENV_SLACK_WEBHOOK_URL",
    "ENV_TEAMS_WEBHOOK_URL",
    "ENV_WEBHOOK_URL",
    "LoggingNotificationAdapter",
    "NotificationAdapter",
    "create_slack_notification_channel",
    "create_teams_notification_channel",
    "create_webhook_notification_channel",
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


def create_teams_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Re-export — composition root is ``integrations.providers.teams``."""
    from intergrax.integrations.providers.teams.bundle import (
        create_teams_notification_channel as _create,
    )

    return _create(webhook_url=webhook_url, delivery=delivery, **config_overrides)


def create_webhook_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    delivery: Optional[NotificationDelivery] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Re-export — composition root is ``integrations.providers.webhook``."""
    from intergrax.integrations.providers.webhook.bundle import (
        create_webhook_notification_channel as _create,
    )

    return _create(webhook_url=webhook_url, delivery=delivery, **config_overrides)


def resolve_notification_adapter(
    channel: Optional[str],
    *,
    delivery: Optional[NotificationDelivery] = None,
) -> NotificationAdapter:
    normalized = (channel or "log").strip().lower()
    if normalized == "slack":
        return create_slack_notification_channel(delivery=delivery)
    if normalized == "teams":
        return create_teams_notification_channel(delivery=delivery)
    if normalized == "webhook":
        return create_webhook_notification_channel(delivery=delivery)
    return _resolve_notification_adapter(channel, delivery=delivery)
