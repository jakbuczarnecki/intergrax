# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "REDPANDA_MESSAGE_BUS_PROVIDER_ID",
    "RedpandaMessageBusIntegration",
    "RedpandaMessageBusIntegrationConfig",
    "RedpandaMessageBusClient",
    "create_redpanda_message_bus",
    "create_redpanda_message_bus_integration",
    "register_redpanda_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_redpanda_message_bus",
        "create_redpanda_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "REDPANDA_MESSAGE_BUS_PROVIDER_ID",
        "RedpandaMessageBusIntegration",
        "RedpandaMessageBusIntegrationConfig",
        "RedpandaMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "REDPANDA_MESSAGE_BUS_PROVIDER_ID",
        "RedpandaMessageBusIntegration",
        "RedpandaMessageBusIntegrationConfig",
        "RedpandaMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_redpanda_integration":
        from intergrax.integrations.providers.message_bus.redpanda.register import register_redpanda_integration

        return register_redpanda_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.redpanda import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.redpanda import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.redpanda import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
