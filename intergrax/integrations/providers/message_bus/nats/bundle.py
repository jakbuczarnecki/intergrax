# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_nats_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.nats.integration import (
    NATS_MESSAGE_BUS_PROVIDER_ID,
    NatsMessageBusIntegration,
    NatsMessageBusIntegrationConfig,
    NatsMessageBusClient,
)

__all__ = [
    "create_nats_message_bus",
    "create_nats_message_bus_integration",
]


def create_nats_message_bus_integration(
    *,
    client: NatsMessageBusClient | None = None,
    enabled: bool = False,
) -> NatsMessageBusIntegration:
    """
    Build a contract-based Nats message bus integration.

    The legacy facade (create_nats_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Nats message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return NatsMessageBusIntegration.from_client(client, enabled=enabled)
    return NatsMessageBusIntegration.for_provider(
        provider_id=NATS_MESSAGE_BUS_PROVIDER_ID,
        display_name="Nats",
        config=NatsMessageBusIntegrationConfig(enabled=enabled),
    )
