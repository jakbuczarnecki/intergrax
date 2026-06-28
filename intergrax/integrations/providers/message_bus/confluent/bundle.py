# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_confluent_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.confluent.integration import (
    CONFLUENT_MESSAGE_BUS_PROVIDER_ID,
    ConfluentMessageBusIntegration,
    ConfluentMessageBusIntegrationConfig,
    ConfluentMessageBusClient,
)

__all__ = [
    "create_confluent_message_bus",
    "create_confluent_message_bus_integration",
]


def create_confluent_message_bus_integration(
    *,
    client: ConfluentMessageBusClient | None = None,
    enabled: bool = False,
) -> ConfluentMessageBusIntegration:
    """
    Build a contract-based Confluent message bus integration.

    The legacy facade (create_confluent_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Confluent message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ConfluentMessageBusIntegration.from_client(client, enabled=enabled)
    return ConfluentMessageBusIntegration.for_provider(
        provider_id=CONFLUENT_MESSAGE_BUS_PROVIDER_ID,
        display_name="Confluent",
        config=ConfluentMessageBusIntegrationConfig(enabled=enabled),
    )
