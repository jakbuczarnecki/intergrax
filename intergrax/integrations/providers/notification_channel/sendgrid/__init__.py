# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "SendgridNotificationChannelIntegration",
    "SendgridNotificationChannelIntegrationConfig",
    "SendgridNotificationChannelClient",
    "create_sendgrid_notification_channel",
    "create_sendgrid_notification_channel_integration",
    "register_sendgrid_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_sendgrid_notification_channel",
        "create_sendgrid_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "SendgridNotificationChannelIntegration",
        "SendgridNotificationChannelIntegrationConfig",
        "SendgridNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "SendgridNotificationChannelIntegration",
        "SendgridNotificationChannelIntegrationConfig",
        "SendgridNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_sendgrid_integration":
        from intergrax.integrations.providers.notification_channel.sendgrid.register import register_sendgrid_integration

        return register_sendgrid_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.sendgrid import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.sendgrid import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.sendgrid import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
