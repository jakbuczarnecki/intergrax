# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Slack openers — internal to the slack integration package.

Only this module may construct Slack notification and interaction adapters
for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.notification_channel.slack.adapter import SlackInteractionAdapter
from intergrax.integrations.providers.notification_channel.slack.config import SlackIntegrationConfig
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery


def open_slack_notification_channel(
    config: SlackIntegrationConfig,
    *,
    implementation: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
) -> NotificationAdapter:
    if implementation is not None:
        return implementation
    from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
    from intergrax.runtime.notifications.formatters import SlackPayloadFormatter

    url = config.webhook_url.strip()
    if not url:
        return LoggingNotificationAdapter()
    return WebhookNotificationAdapter(
        webhook_url=url,
        formatter=SlackPayloadFormatter(),
        delivery=delivery,
        channel="slack",
    )


def open_slack_interaction_surface(
    config: SlackIntegrationConfig,
    *,
    implementation: Optional[InteractionAdapter] = None,
) -> InteractionAdapter:
    del config
    if implementation is not None:
        return implementation
    return SlackInteractionAdapter()
