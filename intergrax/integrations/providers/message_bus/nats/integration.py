# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nats message bus integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, Mapping, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
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
    Single public Nats message bus entrypoint.

    Legacy catalog factory (create_nats_message_bus) delegates to this class.
    """

    config: NatsMessageBusIntegrationConfig = NatsMessageBusIntegrationConfig()
    _client: NatsMessageBusClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> NatsMessageBusIntegration:
        integration = cls.for_provider(
            provider_id=NATS_MESSAGE_BUS_PROVIDER_ID,
            display_name="Nats",
            config=NatsMessageBusIntegrationConfig(enabled=enabled),
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
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)

MessageBus.register(NatsMessageBusIntegration)
