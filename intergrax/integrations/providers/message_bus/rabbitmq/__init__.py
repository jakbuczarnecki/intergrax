# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "RABBITMQ_MESSAGE_BUS_PROVIDER_ID",
    "RabbitmqMessageBusIntegration",
    "RabbitmqMessageBusIntegrationConfig",
    "RabbitmqMessageBusClient",
    "create_rabbitmq_integration",
    "create_rabbitmq_message_bus",
    "create_rabbitmq_message_bus_integration",
    "register_rabbitmq_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_rabbitmq_integration",
        "create_rabbitmq_message_bus",
        "create_rabbitmq_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "RABBITMQ_MESSAGE_BUS_PROVIDER_ID",
        "RabbitmqMessageBusIntegration",
        "RabbitmqMessageBusIntegrationConfig",
        "RabbitmqMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "RABBITMQ_MESSAGE_BUS_PROVIDER_ID",
        "RabbitmqMessageBusIntegration",
        "RabbitmqMessageBusIntegrationConfig",
        "RabbitmqMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_rabbitmq_integration":
        from intergrax.integrations.providers.message_bus.rabbitmq.register import register_rabbitmq_integration

        return register_rabbitmq_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.rabbitmq import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.rabbitmq import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.rabbitmq import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
