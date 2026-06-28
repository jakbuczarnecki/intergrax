# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nats message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NATS_MESSAGE_BUS_PROVIDER_ID = "nats"


class NatsMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Nats message bus integration."""

    pass


@runtime_checkable
class NatsMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class NatsMessageBusIntegration(MessageBusIntegrationContract):
    """
    Nats message bus integration.

    The legacy facade (create_nats_message_bus) remains separate and backward-compatible.
    """

    config: NatsMessageBusIntegrationConfig = NatsMessageBusIntegrationConfig()
    _client: NatsMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: NatsMessageBusClient,
        *,
        enabled: bool = False,
    ) -> NatsMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=NATS_MESSAGE_BUS_PROVIDER_ID,
            display_name="Nats",
            config=NatsMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> NatsMessageBusClient | None:
        return self._client
