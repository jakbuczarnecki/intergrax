# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty integration bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.providers.notification_channel.pagerduty.adapter import _PagerDutyNotificationChannel
from intergrax.integrations.providers.notification_channel.pagerduty.client import PagerDutyEventsClient
from intergrax.integrations.providers.notification_channel.pagerduty.config import PagerDutyIntegrationConfig
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
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[PagerDutyEventsClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PagerDutyIntegrationConfig], Any]] = None,
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


def create_pagerduty_notification_channel(
    *,
    notification_channel: Optional[NotificationChannel] = None,
    client: Optional[PagerDutyEventsClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PagerDutyIntegrationConfig], Any]] = None,
    **config_overrides: object,
) -> PagerdutyNotificationChannelIntegration:
    """Catalog factory for ``"pagerduty"``."""
    return create_pagerduty_integration(
        notification_channel=notification_channel,
        client=client,
        http_client=http_client,
        http_client_factory=http_client_factory,
        **config_overrides,
    ).notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.pagerduty.integration import (
    PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
    PagerdutyNotificationChannelIntegration,
    PagerdutyNotificationChannelIntegrationConfig,
    PagerdutyNotificationChannelClient,
)


def create_pagerduty_notification_channel_integration(
    *,
    client: PagerdutyNotificationChannelClient | None = None,
    enabled: bool = False,
) -> PagerdutyNotificationChannelIntegration:
    """
    Build a contract-based Pagerduty notification channel integration.

    Compatibility shim — constructs Integration via from_store (create_pagerduty_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
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
