# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Service Bus message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID = "service_bus"


class ServiceBusMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Service Bus message bus integration."""

    pass


@runtime_checkable
class ServiceBusMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ServiceBusMessageBusIntegration(MessageBusIntegrationContract):
    """
    Service Bus message bus integration.

    The legacy facade (create_service_bus_message_bus) remains separate and backward-compatible.
    """

    config: ServiceBusMessageBusIntegrationConfig = ServiceBusMessageBusIntegrationConfig()
    _client: ServiceBusMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ServiceBusMessageBusClient,
        *,
        enabled: bool = False,
    ) -> ServiceBusMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID,
            display_name="Service Bus",
            config=ServiceBusMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ServiceBusMessageBusClient | None:
        return self._client
