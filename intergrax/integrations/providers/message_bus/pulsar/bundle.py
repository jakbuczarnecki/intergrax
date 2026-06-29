# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_pulsar_message_bus as _legacy_create_pulsar_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.pulsar.integration import (
    PULSAR_MESSAGE_BUS_PROVIDER_ID,
    PulsarMessageBusIntegration,
    PulsarMessageBusIntegrationConfig,
    PulsarMessageBusClient,
)

__all__ = [
    "create_pulsar_message_bus",
    "create_pulsar_message_bus_integration",
]


def create_pulsar_message_bus_integration(
    *,
    client: PulsarMessageBusClient | None = None,
    enabled: bool = False,
) -> PulsarMessageBusIntegration:
    """
    Build a contract-based Pulsar message bus integration.

    The legacy facade (create_pulsar_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Pulsar message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PulsarMessageBusIntegration.from_client(client, enabled=enabled)
    return PulsarMessageBusIntegration.for_provider(
        provider_id=PULSAR_MESSAGE_BUS_PROVIDER_ID,
        display_name="Pulsar",
        config=PulsarMessageBusIntegrationConfig(enabled=enabled),
    )


def create_pulsar_message_bus(**kwargs: object) -> PulsarMessageBusIntegration:
    """Compatibility shim — constructs PulsarMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_pulsar_message_bus(**kwargs)
    if isinstance(runtime, PulsarMessageBusIntegration):
        return runtime
    return PulsarMessageBusIntegration.from_client(runtime)
