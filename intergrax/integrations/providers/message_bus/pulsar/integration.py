# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pulsar message bus integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence, Mapping

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PULSAR_MESSAGE_BUS_PROVIDER_ID = "pulsar"


class PulsarMessageBusIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Pulsar message bus integration."""

    pass


PulsarMessageBusClient = MessageBus

class PulsarMessageBusIntegration(MessageBusIntegrationContract):
    """
    Single public Pulsar message bus entrypoint.

    Legacy catalog factory (create_pulsar_message_bus) owns catalog behavior; legacy factories use from_client().
    """

    config: PulsarMessageBusIntegrationConfig = PulsarMessageBusIntegrationConfig()
    _client: PulsarMessageBusClient | None = PrivateAttr(default=None)
    


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

MessageBus.register(PulsarMessageBusIntegration)
