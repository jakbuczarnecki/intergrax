# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Grafana Oncall notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID = "grafana_oncall"


class GrafanaOncallNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Grafana Oncall notification channel integration."""

    pass


@runtime_checkable
class GrafanaOncallNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class GrafanaOncallNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Grafana Oncall notification channel integration.

    The legacy facade (create_grafana_oncall_notification_channel) remains separate and backward-compatible.
    """

    config: GrafanaOncallNotificationChannelIntegrationConfig = GrafanaOncallNotificationChannelIntegrationConfig()
    _client: GrafanaOncallNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: GrafanaOncallNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> GrafanaOncallNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Grafana Oncall",
            config=GrafanaOncallNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> GrafanaOncallNotificationChannelClient | None:
        return self._client
