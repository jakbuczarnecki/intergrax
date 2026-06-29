# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_upstash_qstash_message_bus as _legacy_create_upstash_qstash_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.upstash_qstash.integration import (
    UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID,
    UpstashQstashMessageBusIntegration,
    UpstashQstashMessageBusIntegrationConfig,
    UpstashQstashMessageBusClient,
)

__all__ = [
    "create_upstash_qstash_message_bus",
    "create_upstash_qstash_message_bus_integration",
]


def create_upstash_qstash_message_bus_integration(
    *,
    client: UpstashQstashMessageBusClient | None = None,
    enabled: bool = False,
) -> UpstashQstashMessageBusIntegration:
    """
    Build a contract-based Upstash Qstash message bus integration.

    The legacy facade (create_upstash_qstash_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Upstash Qstash message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return UpstashQstashMessageBusIntegration.from_client(client, enabled=enabled)
    return UpstashQstashMessageBusIntegration.for_provider(
        provider_id=UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID,
        display_name="Upstash Qstash",
        config=UpstashQstashMessageBusIntegrationConfig(enabled=enabled),
    )


def create_upstash_qstash_message_bus(**kwargs: object) -> UpstashQstashMessageBusIntegration:
    """Compatibility shim — constructs UpstashQstashMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_upstash_qstash_message_bus(**kwargs)
    if isinstance(runtime, UpstashQstashMessageBusIntegration):
        return runtime
    return UpstashQstashMessageBusIntegration.from_runtime(runtime)
