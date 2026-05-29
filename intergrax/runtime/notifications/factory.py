# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory and configuration for notification adapters (§18, Phase H.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from intergrax.runtime.notifications.adapter_contract import NotificationAdapter
from intergrax.runtime.notifications.adapters.logging_adapter import LoggingNotificationAdapter
from intergrax.runtime.notifications.adapters.webhook_adapter import WebhookNotificationAdapter
from intergrax.runtime.notifications.delivery_contract import NotificationDelivery
from intergrax.runtime.notifications.formatters import (
    GenericJsonPayloadFormatter,
    NotificationPayloadFormatter,
    SlackPayloadFormatter,
    TeamsPayloadFormatter,
)

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


def _formatter_for_backend(backend: NotificationBackend) -> NotificationPayloadFormatter:
    if backend == NotificationBackend.SLACK:
        return SlackPayloadFormatter()
    if backend == NotificationBackend.TEAMS:
        return TeamsPayloadFormatter()
    return GenericJsonPayloadFormatter()


def _webhook_url_for_backend(
    settings: NotificationSettings,
    backend: NotificationBackend,
) -> str:
    if backend == NotificationBackend.SLACK:
        return settings.slack_webhook_url
    if backend == NotificationBackend.TEAMS:
        return settings.teams_webhook_url
    if backend == NotificationBackend.WEBHOOK:
        return settings.webhook_url
    return ""


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
        return LoggingNotificationAdapter()

    if backend == NotificationBackend.SLACK:
        from intergrax.integrations.providers.slack.config import SlackIntegrationConfig
        from intergrax.integrations.providers.slack.opens import open_slack_notification_channel

        config = SlackIntegrationConfig.from_env(webhook_url=resolved.slack_webhook_url)
        return open_slack_notification_channel(config, delivery=delivery)

    url = _webhook_url_for_backend(resolved, backend)
    if not url:
        return LoggingNotificationAdapter()

    return WebhookNotificationAdapter(
        webhook_url=url,
        formatter=formatter or _formatter_for_backend(backend),
        delivery=delivery,
        channel=backend.value,
    )


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
