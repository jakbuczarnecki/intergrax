# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kafka message bus integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

KAFKA_MESSAGE_BUS_PROVIDER_ID = "kafka"


class KafkaMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Kafka message bus integration."""

    pass


@runtime_checkable
class KafkaMessageBusClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class KafkaMessageBusIntegration(MessageBusIntegrationContract):
    """
    Kafka message bus integration.

    The legacy facade (create_kafka_integration) remains separate and backward-compatible.
    """

    config: KafkaMessageBusIntegrationConfig = KafkaMessageBusIntegrationConfig()
    _client: KafkaMessageBusClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: KafkaMessageBusClient,
        *,
        enabled: bool = False,
    ) -> KafkaMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=KAFKA_MESSAGE_BUS_PROVIDER_ID,
            display_name="Kafka",
            config=KafkaMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> KafkaMessageBusClient | None:
        return self._client
