# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Upstash Qstash message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID = "upstash_qstash"


class UpstashQstashMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Upstash Qstash message bus integration."""

    pass


@runtime_checkable
class UpstashQstashMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class UpstashQstashMessageBusIntegration(MessageBusIntegrationContract):
    """
    Upstash Qstash message bus integration.

    The legacy facade (create_upstash_qstash_message_bus) remains separate and backward-compatible.
    """

    config: UpstashQstashMessageBusIntegrationConfig = UpstashQstashMessageBusIntegrationConfig()
    _client: UpstashQstashMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: UpstashQstashMessageBusClient,
        *,
        enabled: bool = False,
    ) -> UpstashQstashMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID,
            display_name="Upstash Qstash",
            config=UpstashQstashMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> UpstashQstashMessageBusClient | None:
        return self._client
