# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PUBSUB_MESSAGE_BUS_PROVIDER_ID",
    "PubsubMessageBusIntegration",
    "PubsubMessageBusIntegrationConfig",
    "PubsubMessageBusClient",
    "create_pubsub_message_bus",
    "create_pubsub_message_bus_integration",
    "register_pubsub_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_pubsub_message_bus",
        "create_pubsub_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PUBSUB_MESSAGE_BUS_PROVIDER_ID",
        "PubsubMessageBusIntegration",
        "PubsubMessageBusIntegrationConfig",
        "PubsubMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PUBSUB_MESSAGE_BUS_PROVIDER_ID",
        "PubsubMessageBusIntegration",
        "PubsubMessageBusIntegrationConfig",
        "PubsubMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_pubsub_integration":
        from intergrax.integrations.providers.message_bus.pubsub.register import register_pubsub_integration

        return register_pubsub_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.pubsub import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.pubsub import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.pubsub import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
