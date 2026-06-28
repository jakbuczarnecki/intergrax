# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Temporal message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TEMPORAL_MESSAGE_BUS_PROVIDER_ID = "temporal"


class TemporalMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Temporal message bus integration."""

    pass


@runtime_checkable
class TemporalMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TemporalMessageBusIntegration(MessageBusIntegrationContract):
    """
    Temporal message bus integration.

    The legacy facade (create_temporal_message_bus) remains separate and backward-compatible.
    """

    config: TemporalMessageBusIntegrationConfig = TemporalMessageBusIntegrationConfig()
    _client: TemporalMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TemporalMessageBusClient,
        *,
        enabled: bool = False,
    ) -> TemporalMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=TEMPORAL_MESSAGE_BUS_PROVIDER_ID,
            display_name="Temporal",
            config=TemporalMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TemporalMessageBusClient | None:
        return self._client
