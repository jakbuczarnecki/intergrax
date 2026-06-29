# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PagerDuty openers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.providers.notification_channel.pagerduty.adapter import _PagerDutyNotificationChannel
from intergrax.integrations.providers.notification_channel.pagerduty.integration import PagerdutyNotificationChannelIntegration
from intergrax.integrations.providers.notification_channel.pagerduty.client import PagerDutyEventsClient
from intergrax.integrations.providers.notification_channel.pagerduty.config import PagerDutyIntegrationConfig


def _create_http_client(config: PagerDutyIntegrationConfig) -> Any:
    import httpx

    timeout = float(config.timeout_seconds or 30.0)
    return httpx.Client(base_url=config.base_url.rstrip("/"), timeout=timeout)


def open_pagerduty_events_client(
    config: PagerDutyIntegrationConfig,
    *,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PagerDutyIntegrationConfig], Any]] = None,
) -> PagerDutyEventsClient:
    if http_client is None:
        factory = http_client_factory or _create_http_client
        http_client = factory(config)
    return PagerDutyEventsClient(config, http_client=http_client)


def open_pagerduty_notification_channel(
    config: PagerDutyIntegrationConfig,
    *,
    implementation: Optional[NotificationChannel] = None,
    client: Optional[PagerDutyEventsClient] = None,
    http_client: Optional[Any] = None,
    http_client_factory: Optional[Callable[[PagerDutyIntegrationConfig], Any]] = None,
) -> NotificationChannel:
    if implementation is not None:
        return implementation
    events_client = client or open_pagerduty_events_client(
        config,
        http_client=http_client,
        http_client_factory=http_client_factory,
    )
    return PagerdutyNotificationChannelIntegration.from_runtime(_PagerDutyNotificationChannel(events_client))