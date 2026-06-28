# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "CONFLUENT_MESSAGE_BUS_PROVIDER_ID",
    "ConfluentMessageBusIntegration",
    "ConfluentMessageBusIntegrationConfig",
    "ConfluentMessageBusClient",
    "create_confluent_message_bus",
    "create_confluent_message_bus_integration",
    "register_confluent_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_confluent_message_bus",
        "create_confluent_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "CONFLUENT_MESSAGE_BUS_PROVIDER_ID",
        "ConfluentMessageBusIntegration",
        "ConfluentMessageBusIntegrationConfig",
        "ConfluentMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "CONFLUENT_MESSAGE_BUS_PROVIDER_ID",
        "ConfluentMessageBusIntegration",
        "ConfluentMessageBusIntegrationConfig",
        "ConfluentMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_confluent_integration":
        from intergrax.integrations.providers.message_bus.confluent.register import register_confluent_integration

        return register_confluent_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.confluent import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.confluent import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.confluent import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
