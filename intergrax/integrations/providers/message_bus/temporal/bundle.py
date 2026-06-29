# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_temporal_message_bus as _legacy_create_temporal_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.temporal.integration import (
    TEMPORAL_MESSAGE_BUS_PROVIDER_ID,
    TemporalMessageBusIntegration,
    TemporalMessageBusIntegrationConfig,
    TemporalMessageBusClient,
)

__all__ = [
    "create_temporal_message_bus",
    "create_temporal_message_bus_integration",
]


def create_temporal_message_bus_integration(
    *,
    client: TemporalMessageBusClient | None = None,
    enabled: bool = False,
) -> TemporalMessageBusIntegration:
    """
    Build a contract-based Temporal message bus integration.

    The legacy facade (create_temporal_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Temporal message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TemporalMessageBusIntegration.from_client(client, enabled=enabled)
    return TemporalMessageBusIntegration.for_provider(
        provider_id=TEMPORAL_MESSAGE_BUS_PROVIDER_ID,
        display_name="Temporal",
        config=TemporalMessageBusIntegrationConfig(enabled=enabled),
    )


def create_temporal_message_bus(**kwargs: object) -> TemporalMessageBusIntegration:
    """Compatibility shim — constructs TemporalMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_temporal_message_bus(**kwargs)
    if isinstance(runtime, TemporalMessageBusIntegration):
        return runtime
    return TemporalMessageBusIntegration.from_client(runtime)
