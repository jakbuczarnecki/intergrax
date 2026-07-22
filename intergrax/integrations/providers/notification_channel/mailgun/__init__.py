# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "MailgunNotificationChannelIntegration",
    "MailgunNotificationChannelIntegrationConfig",
    "MailgunNotificationChannelClient",
    "create_mailgun_notification_channel",
    "create_mailgun_notification_channel_integration",
    "register_mailgun_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_mailgun_notification_channel",
        "create_mailgun_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MAILGUN_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "MailgunNotificationChannelIntegration",
        "MailgunNotificationChannelIntegrationConfig",
        "MailgunNotificationChannelClient",
    }
)


def __getattr__(name: str):
    if name == "register_mailgun_integration":
        from intergrax.integrations.providers.notification_channel.mailgun.register import (
            register_mailgun_integration,
        )

        return register_mailgun_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.mailgun import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.mailgun import (
            integration as _integration,
        )

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
