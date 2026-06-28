# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "OpsgenieNotificationChannelIntegration",
    "OpsgenieNotificationChannelIntegrationConfig",
    "OpsgenieNotificationChannelClient",
    "create_opsgenie_notification_channel",
    "create_opsgenie_notification_channel_integration",
    "register_opsgenie_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_opsgenie_notification_channel",
        "create_opsgenie_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "OpsgenieNotificationChannelIntegration",
        "OpsgenieNotificationChannelIntegrationConfig",
        "OpsgenieNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "OpsgenieNotificationChannelIntegration",
        "OpsgenieNotificationChannelIntegrationConfig",
        "OpsgenieNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_opsgenie_integration":
        from intergrax.integrations.providers.notification_channel.opsgenie.register import register_opsgenie_integration

        return register_opsgenie_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.opsgenie import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.opsgenie import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.opsgenie import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
