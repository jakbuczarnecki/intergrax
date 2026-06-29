# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "NATS_MESSAGE_BUS_PROVIDER_ID",
    "NatsMessageBusIntegration",
    "NatsMessageBusIntegrationConfig",
    "NatsMessageBusClient",
    "create_nats_message_bus",
    "create_nats_message_bus_integration",
    "register_nats_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_nats_message_bus",
        "create_nats_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "NATS_MESSAGE_BUS_PROVIDER_ID",
        "NatsMessageBusIntegration",
        "NatsMessageBusIntegrationConfig",
        "NatsMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "NATS_MESSAGE_BUS_PROVIDER_ID",
        "NatsMessageBusIntegration",
        "NatsMessageBusIntegrationConfig",
        "NatsMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_nats_integration":
        from intergrax.integrations.providers.message_bus.nats.register import register_nats_integration

        return register_nats_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.nats import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.nats import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.nats import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
