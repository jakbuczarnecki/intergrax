# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluent message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

CONFLUENT_MESSAGE_BUS_PROVIDER_ID = "confluent"


class ConfluentMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Confluent message bus integration."""

    pass


@runtime_checkable
class ConfluentMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ConfluentMessageBusIntegration(MessageBusIntegrationContract):
    """
    Confluent message bus integration.

    The legacy facade (create_confluent_message_bus) remains separate and backward-compatible.
    """

    config: ConfluentMessageBusIntegrationConfig = ConfluentMessageBusIntegrationConfig()
    _client: ConfluentMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ConfluentMessageBusClient,
        *,
        enabled: bool = False,
    ) -> ConfluentMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=CONFLUENT_MESSAGE_BUS_PROVIDER_ID,
            display_name="Confluent",
            config=ConfluentMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ConfluentMessageBusClient | None:
        return self._client
