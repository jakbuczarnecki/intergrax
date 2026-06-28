# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pulsar message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PULSAR_MESSAGE_BUS_PROVIDER_ID = "pulsar"


class PulsarMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pulsar message bus integration."""

    pass


@runtime_checkable
class PulsarMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PulsarMessageBusIntegration(MessageBusIntegrationContract):
    """
    Pulsar message bus integration.

    The legacy facade (create_pulsar_message_bus) remains separate and backward-compatible.
    """

    config: PulsarMessageBusIntegrationConfig = PulsarMessageBusIntegrationConfig()
    _client: PulsarMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: PulsarMessageBusClient,
        *,
        enabled: bool = False,
    ) -> PulsarMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=PULSAR_MESSAGE_BUS_PROVIDER_ID,
            display_name="Pulsar",
            config=PulsarMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PulsarMessageBusClient | None:
        return self._client
