# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opsgenie notification channel integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import NotificationChannelIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID = "opsgenie"


class OpsgenieNotificationChannelIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Opsgenie notification channel integration."""

    pass


@runtime_checkable
class OpsgenieNotificationChannelClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class OpsgenieNotificationChannelIntegration(NotificationChannelIntegrationContract):
    """
    Opsgenie notification channel integration.

    The legacy facade (create_opsgenie_notification_channel) remains separate and backward-compatible.
    """

    config: OpsgenieNotificationChannelIntegrationConfig = OpsgenieNotificationChannelIntegrationConfig()
    _client: OpsgenieNotificationChannelClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: OpsgenieNotificationChannelClient,
        *,
        enabled: bool = False,
    ) -> OpsgenieNotificationChannelIntegration:
        integration = cls.for_provider(
            provider_id=OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
            display_name="Opsgenie",
            config=OpsgenieNotificationChannelIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> OpsgenieNotificationChannelClient | None:
        return self._client
