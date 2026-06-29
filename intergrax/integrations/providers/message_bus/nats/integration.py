# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nats message bus integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

NATS_MESSAGE_BUS_PROVIDER_ID = "nats"


class NatsMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Nats message bus integration."""

    pass


NatsMessageBusClient = MessageBus

class NatsMessageBusIntegration(MessageBusIntegrationContract):
    """
    Single public Nats message bus entrypoint.

    Legacy catalog factory (create_nats_message_bus) owns catalog behavior; legacy factories use from_client().
    """

    config: NatsMessageBusIntegrationConfig = NatsMessageBusIntegrationConfig()
    _client: NatsMessageBusClient | None = PrivateAttr(default=None)
    


    def publish(self, topic: str, payload: bytes, *, headers: Mapping[str, str] | None = None) -> None:
        self._require_client().publish(topic, payload, headers=headers)

    def close(self) -> None:
        self._require_client().close()


    def cancel(self, handle):
        return self._require_client().cancel(handle)

    def enqueue(self, request):
        return self._require_client().enqueue(request)

    def get_result(self, handle):
        return self._require_client().get_result(handle)

    def get_status(self, handle):
        return self._require_client().get_status(handle)

    def list_tasks(self, tenant_id, limit: int = 50, status_filter: Optional[TaskStatus] = None):
        return self._require_client().list_tasks(tenant_id, limit=limit, status_filter=status_filter)

    def purge_completed(self, tenant_id, older_than_seconds: int = 0):
        return self._require_client().purge_completed(tenant_id, older_than_seconds=older_than_seconds)

    def _require_client(self) -> MessageBus:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

MessageBus.register(NatsMessageBusIntegration)
