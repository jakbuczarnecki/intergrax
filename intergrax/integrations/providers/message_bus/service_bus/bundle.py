# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_service_bus_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.service_bus.integration import (
    SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID,
    ServiceBusMessageBusIntegration,
    ServiceBusMessageBusIntegrationConfig,
    ServiceBusMessageBusClient,
)

__all__ = [
    "create_service_bus_message_bus",
    "create_service_bus_message_bus_integration",
]


def create_service_bus_message_bus_integration(
    *,
    client: ServiceBusMessageBusClient | None = None,
    enabled: bool = False,
) -> ServiceBusMessageBusIntegration:
    """
    Build a contract-based Service Bus message bus integration.

    The legacy facade (create_service_bus_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Service Bus message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return ServiceBusMessageBusIntegration.from_client(client, enabled=enabled)
    return ServiceBusMessageBusIntegration.for_provider(
        provider_id=SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID,
        display_name="Service Bus",
        config=ServiceBusMessageBusIntegrationConfig(enabled=enabled),
    )
