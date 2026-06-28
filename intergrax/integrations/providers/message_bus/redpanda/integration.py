# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redpanda message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

REDPANDA_MESSAGE_BUS_PROVIDER_ID = "redpanda"


class RedpandaMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Redpanda message bus integration."""

    pass


@runtime_checkable
class RedpandaMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class RedpandaMessageBusIntegration(MessageBusIntegrationContract):
    """
    Redpanda message bus integration.

    The legacy facade (create_redpanda_message_bus) remains separate and backward-compatible.
    """

    config: RedpandaMessageBusIntegrationConfig = RedpandaMessageBusIntegrationConfig()
    _client: RedpandaMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: RedpandaMessageBusClient,
        *,
        enabled: bool = False,
    ) -> RedpandaMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=REDPANDA_MESSAGE_BUS_PROVIDER_ID,
            display_name="Redpanda",
            config=RedpandaMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> RedpandaMessageBusClient | None:
        return self._client
