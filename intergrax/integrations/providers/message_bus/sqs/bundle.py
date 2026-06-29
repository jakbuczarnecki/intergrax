# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_sqs_message_bus as _legacy_create_sqs_message_bus

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.sqs.integration import (
    SQS_MESSAGE_BUS_PROVIDER_ID,
    SqsMessageBusIntegration,
    SqsMessageBusIntegrationConfig,
    SqsMessageBusClient,
)

__all__ = [
    "create_sqs_message_bus",
    "create_sqs_message_bus_integration",
]


def create_sqs_message_bus_integration(
    *,
    client: SqsMessageBusClient | None = None,
    enabled: bool = False,
) -> SqsMessageBusIntegration:
    """
    Build a contract-based Sqs message bus integration.

    The legacy facade (create_sqs_message_bus) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Sqs message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SqsMessageBusIntegration.from_client(client, enabled=enabled)
    return SqsMessageBusIntegration.for_provider(
        provider_id=SQS_MESSAGE_BUS_PROVIDER_ID,
        display_name="Sqs",
        config=SqsMessageBusIntegrationConfig(enabled=enabled),
    )


def create_sqs_message_bus(**kwargs: object) -> SqsMessageBusIntegration:
    """Compatibility shim — constructs SqsMessageBusIntegration from legacy runtime."""
    runtime = _legacy_create_sqs_message_bus(**kwargs)
    if isinstance(runtime, SqsMessageBusIntegration):
        return runtime
    return SqsMessageBusIntegration.from_client(runtime)
