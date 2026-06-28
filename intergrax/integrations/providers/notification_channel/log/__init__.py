# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LOG_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "LogNotificationChannelIntegration",
    "LogNotificationChannelIntegrationConfig",
    "LogNotificationChannelClient",
    "create_log_integration",
    "create_log_notification_channel",
    "create_log_notification_channel_integration",
    "register_log_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_log_integration",
        "create_log_notification_channel",
        "create_log_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LOG_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "LogNotificationChannelIntegration",
        "LogNotificationChannelIntegrationConfig",
        "LogNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "LOG_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "LogNotificationChannelIntegration",
        "LogNotificationChannelIntegrationConfig",
        "LogNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_log_integration":
        from intergrax.integrations.providers.notification_channel.log.register import register_log_integration

        return register_log_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.log import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.log import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.log import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
