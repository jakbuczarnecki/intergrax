# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Kafka message bus integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
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
    Single public Kafka message bus entrypoint.

    Legacy catalog factory (create_kafka_integration) delegates to this class.
    """

    config: KafkaMessageBusIntegrationConfig = KafkaMessageBusIntegrationConfig()
    _client: KafkaMessageBusClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> KafkaMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=KAFKA_MESSAGE_BUS_PROVIDER_ID,
            display_name="Kafka",
            config=KafkaMessageBusIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def publish(self, topic: str, payload: bytes, *, headers: Mapping[str, str] | None = None) -> None:
        self._require_runtime().publish(topic, payload, headers=headers)

    def close(self) -> None:
        self._require_runtime().close()


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

MessageBus.register(KafkaMessageBusIntegration)
