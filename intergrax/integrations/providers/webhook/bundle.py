# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete generic webhook integration bundle — composition root for HTTP outbound notifications.

All runtime wiring MUST use this module or
``profile.resolve(IntegrationCategory.NOTIFICATION_CHANNEL)`` with ``IntegrationSlug.WEBHOOK``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.integrations.providers.webhook.config import WebhookIntegrationConfig
from intergrax.integrations.providers.webhook.opens import open_webhook_notification_channel
from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.formatters import NotificationPayloadFormatter


@dataclass(frozen=True)
class WebhookIntegrationBundle:
    """Generic webhook notification adapter + config."""

    config: WebhookIntegrationConfig
    notification_channel: NotificationAdapter


def resolve_webhook_config(**overrides: object) -> WebhookIntegrationConfig:
    return WebhookIntegrationConfig.from_env(**overrides)


def create_webhook_integration(
    *,
    webhook_url: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    formatter: Optional[NotificationPayloadFormatter] = None,
    **config_overrides: object,
) -> WebhookIntegrationBundle:
    """Single entry point for generic HTTP webhook notifications."""
    overrides: dict[str, object] = dict(config_overrides)
    if webhook_url is not None:
        overrides["webhook_url"] = webhook_url

    config = resolve_webhook_config(**overrides)
    notification = open_webhook_notification_channel(
        config,
        implementation=notification_adapter,
        delivery=delivery,
        formatter=formatter,
    )

    return WebhookIntegrationBundle(
        config=config,
        notification_channel=notification,
    )


def create_webhook_notification_channel(
    *,
    webhook_url: Optional[str] = None,
    notification_adapter: Optional[NotificationAdapter] = None,
    delivery: Optional[NotificationDelivery] = None,
    formatter: Optional[NotificationPayloadFormatter] = None,
    **config_overrides: object,
) -> NotificationAdapter:
    """Catalog factory for ``IntegrationSlug.WEBHOOK`` / ``NOTIFICATION_CHANNEL``."""
    return create_webhook_integration(
        webhook_url=webhook_url,
        notification_adapter=notification_adapter,
        delivery=delivery,
        formatter=formatter,
        **config_overrides,
    ).notification_channel
