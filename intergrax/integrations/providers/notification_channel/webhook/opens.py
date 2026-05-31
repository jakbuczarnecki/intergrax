# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level generic webhook openers — internal to the webhook integration package.

Only this module may construct ``WebhookNotificationAdapter`` instances for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.notification_channel.webhook.config import WebhookIntegrationConfig
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.formatters import GenericJsonPayloadFormatter, NotificationPayloadFormatter


def open_webhook_notification_channel(
    config: WebhookIntegrationConfig,
    *,
    implementation: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    formatter: Optional[NotificationPayloadFormatter] = None,
) -> NotificationAdapter:
    if implementation is not None:
        return implementation
    from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter

    url = config.webhook_url.strip()
    if not url:
        return LoggingNotificationAdapter()
    return WebhookNotificationAdapter(
        webhook_url=url,
        formatter=formatter or GenericJsonPayloadFormatter(),
        delivery=delivery,
        channel="webhook",
    )
