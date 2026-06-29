# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "MailpitNotificationChannelIntegration",
    "MailpitNotificationChannelIntegrationConfig",
    "MailpitNotificationChannelClient",
    "create_mailpit_notification_channel",
    "create_mailpit_notification_channel_integration",
    "register_mailpit_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_mailpit_notification_channel",
        "create_mailpit_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "MailpitNotificationChannelIntegration",
        "MailpitNotificationChannelIntegrationConfig",
        "MailpitNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "MailpitNotificationChannelIntegration",
        "MailpitNotificationChannelIntegrationConfig",
        "MailpitNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_mailpit_integration":
        from intergrax.integrations.providers.notification_channel.mailpit.register import register_mailpit_integration

        return register_mailpit_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.mailpit import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.mailpit import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.mailpit import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
