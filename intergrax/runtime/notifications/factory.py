# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory and configuration for notification adapters (§18, Phase H.1)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
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
    settings: NotificationSettings | None = None,
    *,
    implementation: NotificationAdapter | None = None,
    factory: NotificationAdapterFactory | None = None,
    delivery: NotificationDelivery | None = None,
    formatter: NotificationPayloadFormatter | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> NotificationAdapter:
    """
    Build a notification adapter.

    Priority: explicit ``implementation`` > ``factory`` > ``settings``/env defaults.
    """
    if implementation is not None:
        return implementation
    if factory is not None:
        return factory()

    resolved = settings or resolve_notification_settings(lab_opt_in=lab_opt_in)
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
        from intergrax.integrations.providers.notification_channel.webhook.opens import (
            open_webhook_notification_channel,
        )

        config = WebhookIntegrationConfig.from_env(webhook_url=resolved.webhook_url)
        return open_webhook_notification_channel(config, delivery=delivery, formatter=formatter)

    if backend == NotificationBackend.PAGERDUTY:
        from intergrax.integrations.providers.notification_channel.pagerduty.bundle import (
            create_pagerduty_notification_channel,
        )

        return create_pagerduty_notification_channel()

    if backend == NotificationBackend.OPSGENIE:
        from intergrax.integrations.providers.notification_channel.opsgenie.bundle import (
            create_opsgenie_notification_channel,
        )

        return create_opsgenie_notification_channel()

    raise IntegrationConfigurationError(f"unsupported notification backend {backend!r}")


def resolve_notification_adapter(
    channel: str | None,
    *,
    settings: NotificationSettings | None = None,
    delivery: NotificationDelivery | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> NotificationAdapter:
    """
    Resolve adapter by task notify channel (long-running / HITL).
    """
    if channel is None or not str(channel).strip():
        if lab_opt_in is None:
            raise IntegrationConfigurationError("notify_channel is unset and no notification adapter was configured")
        return create_notification_adapter(
            resolve_notification_settings(lab_opt_in=lab_opt_in),
            delivery=delivery,
        )

    normalized = str(channel).strip().lower()
    try:
        backend = NotificationBackend(normalized)
    except ValueError as exc:
        raise IntegrationConfigurationError(f"unknown notification channel {normalized!r}") from exc

    base = settings or resolve_notification_settings(lab_opt_in=lab_opt_in)
    merged = NotificationSettings(
        backend=backend,
        webhook_url=base.webhook_url,
        slack_webhook_url=base.slack_webhook_url,
        teams_webhook_url=base.teams_webhook_url,
    )
    return create_notification_adapter(merged, delivery=delivery)
