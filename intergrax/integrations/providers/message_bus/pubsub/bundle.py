# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_pubsub_message_bus as _legacy_create_pubsub_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.pubsub.integration import (
    PUBSUB_MESSAGE_BUS_PROVIDER_ID,
    PubsubMessageBusIntegration,
    PubsubMessageBusIntegrationConfig,
    PubsubMessageBusClient,
)

__all__ = [
    "create_pubsub_message_bus",
    "create_pubsub_message_bus_integration",
]


def create_pubsub_message_bus_integration(
    *,
    client: PubsubMessageBusClient | None = None,
    enabled: bool = False,
) -> PubsubMessageBusIntegration:
    """
    Build a contract-based Pubsub message bus integration.

    The legacy facade (create_pubsub_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Pubsub message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PubsubMessageBusIntegration.from_client(client, enabled=enabled)
    return PubsubMessageBusIntegration.for_provider(
        provider_id=PUBSUB_MESSAGE_BUS_PROVIDER_ID,
        display_name="Pubsub",
        config=PubsubMessageBusIntegrationConfig(enabled=enabled),
    )


def create_pubsub_message_bus(**kwargs: object) -> PubsubMessageBusIntegration:
    """Compatibility shim — constructs PubsubMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_pubsub_message_bus(**kwargs)
    if isinstance(runtime, PubsubMessageBusIntegration):
        return runtime
    return PubsubMessageBusIntegration.from_runtime(runtime)
