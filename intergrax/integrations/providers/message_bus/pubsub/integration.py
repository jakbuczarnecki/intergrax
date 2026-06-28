# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pubsub message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PUBSUB_MESSAGE_BUS_PROVIDER_ID = "pubsub"


class PubsubMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pubsub message bus integration."""

    pass


@runtime_checkable
class PubsubMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PubsubMessageBusIntegration(MessageBusIntegrationContract):
    """
    Pubsub message bus integration.

    The legacy facade (create_pubsub_message_bus) remains separate and backward-compatible.
    """

    config: PubsubMessageBusIntegrationConfig = PubsubMessageBusIntegrationConfig()
    _client: PubsubMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PubsubMessageBusClient,
        *,
        enabled: bool = False,
    ) -> PubsubMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=PUBSUB_MESSAGE_BUS_PROVIDER_ID,
            display_name="Pubsub",
            config=PubsubMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PubsubMessageBusClient | None:
        return self._client
