# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory and configuration for notification adapters (§18, Phase H.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.formatters import NotificationPayloadFormatter

ENV_NOTIFICATION_BACKEND = "INTERGRAX_NOTIFICATION_BACKEND"
ENV_WEBHOOK_URL = "INTERGRAX_WEBHOOK_URL"
ENV_SLACK_WEBHOOK_URL = "INTERGRAX_SLACK_WEBHOOK_URL"
ENV_TEAMS_WEBHOOK_URL = "INTERGRAX_TEAMS_WEBHOOK_URL"

NotificationAdapterFactory = Callable[[], NotificationAdapter]


class NotificationBackend(str, Enum):
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"


@dataclass(frozen=True)
class NotificationSettings:
    backend: NotificationBackend = NotificationBackend.LOG
    webhook_url: str = ""
    slack_webhook_url: str = ""
    teams_webhook_url: str = ""


def resolve_notification_settings(
    *,
    backend: Optional[str] = None,
    webhook_url: Optional[str] = None,
    slack_webhook_url: Optional[str] = None,
    teams_webhook_url: Optional[str] = None,
) -> NotificationSettings:
    raw_backend = (
        backend
        or os.environ.get(ENV_NOTIFICATION_BACKEND, NotificationBackend.LOG.value)
    ).strip().lower()
    try:
        resolved_backend = NotificationBackend(raw_backend)
    except ValueError:
        resolved_backend = NotificationBackend.LOG
    return NotificationSettings(
        backend=resolved_backend,
        webhook_url=(webhook_url or os.environ.get(ENV_WEBHOOK_URL, "")).strip(),
        slack_webhook_url=(
            slack_webhook_url or os.environ.get(ENV_SLACK_WEBHOOK_URL, "")
        ).strip(),
        teams_webhook_url=(
            teams_webhook_url or os.environ.get(ENV_TEAMS_WEBHOOK_URL, "")
        ).strip(),
    )


def create_notification_adapter(
    settings: Optional[NotificationSettings] = None,
    *,
    implementation: Optional[NotificationAdapter] = None,
    factory: Optional[NotificationAdapterFactory] = None,
    delivery: Optional[NotificationDelivery] = None,
    formatter: Optional[NotificationPayloadFormatter] = None,
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
        from intergrax.integrations.providers.notification_channel.slack.config import SlackIntegrationConfig
        from intergrax.integrations.providers.notification_channel.slack.opens import open_slack_notification_channel

        config = SlackIntegrationConfig.from_env(webhook_url=resolved.slack_webhook_url)
        return open_slack_notification_channel(config, delivery=delivery)

    if backend == NotificationBackend.TEAMS:
        from intergrax.integrations.providers.notification_channel.teams.config import TeamsIntegrationConfig
        from intergrax.integrations.providers.notification_channel.teams.opens import open_teams_notification_channel

        config = TeamsIntegrationConfig.from_env(webhook_url=resolved.teams_webhook_url)
        return open_teams_notification_channel(config, delivery=delivery)

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
