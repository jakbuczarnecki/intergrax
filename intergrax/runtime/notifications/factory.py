# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory and configuration for notification adapters (§18, Phase H.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.backend_contract import (
    ENV_NOTIFICATION_BACKEND,
    ENV_SLACK_WEBHOOK_URL,
    ENV_TEAMS_WEBHOOK_URL,
    ENV_WEBHOOK_URL,
    NotificationBackend,
    NotificationSettings,
    resolve_notification_settings,
)
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery

if TYPE_CHECKING:
    from intergrax.runtime.notifications.formatters import NotificationPayloadFormatter

NotificationAdapterFactory = Callable[[], NotificationAdapter]

__all__ = [
    "ENV_NOTIFICATION_BACKEND",
    "ENV_SLACK_WEBHOOK_URL",
    "ENV_TEAMS_WEBHOOK_URL",
    "ENV_WEBHOOK_URL",
    "NotificationAdapterFactory",
    "NotificationBackend",
    "NotificationSettings",
    "create_notification_adapter",
    "resolve_notification_adapter",
    "resolve_notification_settings",
]


def create_notification_adapter(
    settings: Optional[NotificationSettings] = None,
    *,
    implementation: Optional[NotificationAdapter] = None,
    factory: Optional[NotificationAdapterFactory] = None,
    delivery: Optional[NotificationDelivery] = None,
    formatter: "NotificationPayloadFormatter | None" = None,
) -> NotificationAdapter:
    """
    Build a notification adapter.

    Priority: explicit ``implementation`` > ``factory`` > ``settings``/env defaults.
    """
    if implementation is not None:
        return implementation
    if factory is not None:
        return factory()

    resolved = settings or resolve_notification_settings()
    backend = resolved.backend
    if backend == NotificationBackend.LOG:
        from intergrax.integrations.providers.notification_channel.log.bundle import create_log_notification_channel

        return create_log_notification_channel()

    if backend == NotificationBackend.SLACK:
        from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_notification_channel

        return create_slack_notification_channel(
            webhook_url=resolved.slack_webhook_url,
            delivery=delivery,
        )

    if backend == NotificationBackend.TEAMS:
        from intergrax.integrations.providers.notification_channel.teams.bundle import create_teams_notification_channel

        return create_teams_notification_channel(
            webhook_url=resolved.teams_webhook_url,
            delivery=delivery,
        )

    if backend == NotificationBackend.WEBHOOK:
        from intergrax.integrations.providers.notification_channel.webhook.config import WebhookIntegrationConfig
        from intergrax.integrations.providers.notification_channel.webhook.opens import open_webhook_notification_channel

        config = WebhookIntegrationConfig.from_env(webhook_url=resolved.webhook_url)
        return open_webhook_notification_channel(config, delivery=delivery, formatter=formatter)

    if backend == NotificationBackend.PAGERDUTY:
        from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel

        return create_pagerduty_notification_channel()

    if backend == NotificationBackend.OPSGENIE:
        from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel

        return create_opsgenie_notification_channel()
    return create_log_notification_channel()


def resolve_notification_adapter(
    channel: Optional[str],
    *,
    settings: Optional[NotificationSettings] = None,
    delivery: Optional[NotificationDelivery] = None,
) -> NotificationAdapter:
    """
    Resolve adapter by task notify channel (long-running / HITL).

    Channel names map to backends; falls back to log when URL is not configured.
    """
    normalized = (channel or "log").strip().lower()
    try:
        backend = NotificationBackend(normalized)
    except ValueError:
        backend = NotificationBackend.LOG

    base = settings or resolve_notification_settings()
    merged = NotificationSettings(
        backend=backend,
        webhook_url=base.webhook_url,
        slack_webhook_url=base.slack_webhook_url,
        teams_webhook_url=base.teams_webhook_url,
    )
    return create_notification_adapter(merged, delivery=delivery)
