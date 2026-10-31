# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty integration bundle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.providers.notification_channel.pagerduty.client import PagerDutyEventsClient
from intergrax.integrations.providers.notification_channel.pagerduty.config import PagerDutyIntegrationConfig
from intergrax.integrations.providers.notification_channel.pagerduty.integration import (
    PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
    PagerdutyNotificationChannelClient,
    PagerdutyNotificationChannelIntegration,
    PagerdutyNotificationChannelIntegrationConfig,
)
from intergrax.integrations.providers.notification_channel.pagerduty.opens import (
    open_pagerduty_events_client,
    open_pagerduty_notification_channel,
)


@dataclass(frozen=True)
class PagerDutyIntegrationBundle:
    config: PagerDutyIntegrationConfig
    notification_channel: PagerdutyNotificationChannelIntegration
    events_client: PagerDutyEventsClient


def create_pagerduty_integration(
    *,
    notification_channel: NotificationChannel | None = None,
    client: PagerDutyEventsClient | None = None,
    http_client: Any | None = None,
    http_client_factory: Callable[[PagerDutyIntegrationConfig], Any] | None = None,
    **config_overrides: object,
) -> PagerDutyIntegrationBundle:
    config = PagerDutyIntegrationConfig.from_env(**config_overrides)
    events_client = client or open_pagerduty_events_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    channel = open_pagerduty_notification_channel(
        config,
        implementation=notification_channel,
        client=events_client,
    )
    assert isinstance(channel, PagerdutyNotificationChannelIntegration)
    return PagerDutyIntegrationBundle(
        config=config,
        notification_channel=channel,
        events_client=events_client,
    )


def create_pagerduty_notification_channel_integration(
    *,
    client: PagerdutyNotificationChannelClient | None = None,
    enabled: bool = False,
) -> PagerdutyNotificationChannelIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Pagerduty notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PagerdutyNotificationChannelIntegration.from_client(client, enabled=enabled)
    return PagerdutyNotificationChannelIntegration.for_provider(
        provider_id=PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Pagerduty",
        config=PagerdutyNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_pagerduty_notification_channel(
    *,
    notification_channel: NotificationChannel | None = None,
    client: PagerDutyEventsClient | None = None,
    http_client: Any | None = None,
    http_client_factory: Callable[[PagerDutyIntegrationConfig], Any] | None = None,
    **config_overrides: object,
) -> PagerdutyNotificationChannelIntegration:
    """Catalog factory for ``"pagerduty"``."""
    if (
        notification_channel is None
        and client is None
        and http_client is None
        and http_client_factory is None
        and not config_overrides
    ):
        return create_pagerduty_notification_channel_integration(enabled=False)
    return create_pagerduty_integration(
        notification_channel=notification_channel,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).notification_channel
