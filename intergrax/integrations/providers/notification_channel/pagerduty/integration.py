# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pagerduty notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID = "pagerduty"


class PagerdutyNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pagerduty notification channel integration."""

    pass


@runtime_checkable
class PagerdutyNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PagerdutyNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Pagerduty notification channel integration.

    The legacy facade (create_pagerduty_integration) remains separate and backward-compatible.
    """

    config: PagerdutyNotificationChannelIntegrationConfig = PagerdutyNotificationChannelIntegrationConfig()
    _client: PagerdutyNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PagerdutyNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> PagerdutyNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Pagerduty",
            config=PagerdutyNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PagerdutyNotificationChannelClient | None:
        return self._client
