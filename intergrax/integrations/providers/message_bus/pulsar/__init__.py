# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PULSAR_MESSAGE_BUS_PROVIDER_ID",
    "PulsarMessageBusIntegration",
    "PulsarMessageBusIntegrationConfig",
    "PulsarMessageBusClient",
    "create_pulsar_message_bus",
    "create_pulsar_message_bus_integration",
    "register_pulsar_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_pulsar_message_bus",
        "create_pulsar_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PULSAR_MESSAGE_BUS_PROVIDER_ID",
        "PulsarMessageBusIntegration",
        "PulsarMessageBusIntegrationConfig",
        "PulsarMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PULSAR_MESSAGE_BUS_PROVIDER_ID",
        "PulsarMessageBusIntegration",
        "PulsarMessageBusIntegrationConfig",
        "PulsarMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_pulsar_integration":
        from intergrax.integrations.providers.message_bus.pulsar.register import register_pulsar_integration

        return register_pulsar_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.pulsar import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.pulsar import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.pulsar import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
