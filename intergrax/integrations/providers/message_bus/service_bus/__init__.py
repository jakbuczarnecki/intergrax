# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID",
    "ServiceBusMessageBusIntegration",
    "ServiceBusMessageBusIntegrationConfig",
    "ServiceBusMessageBusClient",
    "create_service_bus_message_bus",
    "create_service_bus_message_bus_integration",
    "register_service_bus_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_service_bus_message_bus",
        "create_service_bus_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID",
        "ServiceBusMessageBusIntegration",
        "ServiceBusMessageBusIntegrationConfig",
        "ServiceBusMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SERVICE_BUS_MESSAGE_BUS_PROVIDER_ID",
        "ServiceBusMessageBusIntegration",
        "ServiceBusMessageBusIntegrationConfig",
        "ServiceBusMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_service_bus_integration":
        from intergrax.integrations.providers.message_bus.service_bus.register import register_service_bus_integration

        return register_service_bus_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.service_bus import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.service_bus import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.service_bus import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
