# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level Teams openers — internal to the teams integration package.

Only this module may construct Teams notification and interaction adapters
for catalog wiring.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.providers.notification_channel.teams.adapter import _TeamsInteractionAdapter
from intergrax.integrations.providers.notification_channel.teams.config import TeamsIntegrationConfig
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery


def open_teams_notification_channel(
    config: TeamsIntegrationConfig,
    *,
    implementation: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
) -> NotificationAdapter:
    if implementation is not None:
        return implementation
    from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
    from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
    from intergrax.runtime.notifications.formatters import TeamsPayloadFormatter

    url = config.webhook_url.strip()
    if not url:
        return LoggingNotificationAdapter()
    return WebhookNotificationAdapter(
        webhook_url=url,
        formatter=TeamsPayloadFormatter(),
        delivery=delivery,
        channel="teams",
    )


def open_teams_interaction_surface(
    config: TeamsIntegrationConfig,
    *,
    implementation: Optional[InteractionAdapter] = None,
) -> InteractionAdapter:
    del config
    if implementation is not None:
        return implementation
    return _TeamsInteractionAdapter()
