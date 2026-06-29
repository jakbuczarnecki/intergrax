# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_redpanda_message_bus as _legacy_create_redpanda_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.redpanda.integration import (
    REDPANDA_MESSAGE_BUS_PROVIDER_ID,
    RedpandaMessageBusIntegration,
    RedpandaMessageBusIntegrationConfig,
    RedpandaMessageBusClient,
)

__all__ = [
    "create_redpanda_message_bus",
    "create_redpanda_message_bus_integration",
]


def create_redpanda_message_bus_integration(
    *,
    client: RedpandaMessageBusClient | None = None,
    enabled: bool = False,
) -> RedpandaMessageBusIntegration:
    """
    Build a contract-based Redpanda message bus integration.

    The legacy facade (create_redpanda_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Redpanda message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return RedpandaMessageBusIntegration.from_client(client, enabled=enabled)
    return RedpandaMessageBusIntegration.for_provider(
        provider_id=REDPANDA_MESSAGE_BUS_PROVIDER_ID,
        display_name="Redpanda",
        config=RedpandaMessageBusIntegrationConfig(enabled=enabled),
    )


def create_redpanda_message_bus(**kwargs: object) -> RedpandaMessageBusIntegration:
    """Compatibility shim — constructs RedpandaMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_redpanda_message_bus(**kwargs)
    if isinstance(runtime, RedpandaMessageBusIntegration):
        return runtime
    return RedpandaMessageBusIntegration.from_runtime(runtime)
