# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID",
    "UpstashQstashMessageBusIntegration",
    "UpstashQstashMessageBusIntegrationConfig",
    "UpstashQstashMessageBusClient",
    "create_upstash_qstash_message_bus",
    "create_upstash_qstash_message_bus_integration",
    "register_upstash_qstash_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_upstash_qstash_message_bus",
        "create_upstash_qstash_message_bus_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID",
        "UpstashQstashMessageBusIntegration",
        "UpstashQstashMessageBusIntegrationConfig",
        "UpstashQstashMessageBusClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "UPSTASH_QSTASH_MESSAGE_BUS_PROVIDER_ID",
        "UpstashQstashMessageBusIntegration",
        "UpstashQstashMessageBusIntegrationConfig",
        "UpstashQstashMessageBusClient",
    }
)

def __getattr__(name: str):
    if name == "register_upstash_qstash_integration":
        from intergrax.integrations.providers.message_bus.upstash_qstash.register import register_upstash_qstash_integration

        return register_upstash_qstash_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.message_bus.upstash_qstash import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.upstash_qstash import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.message_bus.upstash_qstash import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
