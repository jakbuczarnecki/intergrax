# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "TEMPORAL_MESSAGE_BUS_PROVIDER_ID",
    "TemporalMessageBusIntegration",
    "TemporalMessageBusIntegrationConfig",
    "TemporalMessageBusClient",
    "create_temporal_message_bus",
    "create_temporal_message_bus_integration",
    "register_temporal_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_temporal_message_bus",
        "create_temporal_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "TEMPORAL_MESSAGE_BUS_PROVIDER_ID",
        "TemporalMessageBusIntegration",
        "TemporalMessageBusIntegrationConfig",
        "TemporalMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "TEMPORAL_MESSAGE_BUS_PROVIDER_ID",
        "TemporalMessageBusIntegration",
        "TemporalMessageBusIntegrationConfig",
        "TemporalMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_temporal_integration":
        from intergrax.integrations.providers.message_bus.temporal.register import register_temporal_integration

        return register_temporal_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.temporal import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.temporal import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.temporal import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
