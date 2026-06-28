# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Rabbitmq message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

RABBITMQ_MESSAGE_BUS_PROVIDER_ID = "rabbitmq"


class RabbitmqMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Rabbitmq message bus integration."""

    pass


@runtime_checkable
class RabbitmqMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class RabbitmqMessageBusIntegration(MessageBusIntegrationContract):
    """
    Rabbitmq message bus integration.

    The legacy facade (create_rabbitmq_integration) remains separate and backward-compatible.
    """

    config: RabbitmqMessageBusIntegrationConfig = RabbitmqMessageBusIntegrationConfig()
    _client: RabbitmqMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: RabbitmqMessageBusClient,
        *,
        enabled: bool = False,
    ) -> RabbitmqMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=RABBITMQ_MESSAGE_BUS_PROVIDER_ID,
            display_name="Rabbitmq",
            config=RabbitmqMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> RabbitmqMessageBusClient | None:
        return self._client
