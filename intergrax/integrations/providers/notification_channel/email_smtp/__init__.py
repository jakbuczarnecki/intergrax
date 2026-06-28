# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "EmailSmtpNotificationChannelIntegration",
    "EmailSmtpNotificationChannelIntegrationConfig",
    "EmailSmtpNotificationChannelClient",
    "create_email_smtp_notification_channel",
    "create_email_smtp_notification_channel_integration",
    "register_email_smtp_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_email_smtp_notification_channel",
        "create_email_smtp_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "EmailSmtpNotificationChannelIntegration",
        "EmailSmtpNotificationChannelIntegrationConfig",
        "EmailSmtpNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "EMAIL_SMTP_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "EmailSmtpNotificationChannelIntegration",
        "EmailSmtpNotificationChannelIntegrationConfig",
        "EmailSmtpNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_email_smtp_integration":
        from intergrax.integrations.providers.notification_channel.email_smtp.register import register_email_smtp_integration

        return register_email_smtp_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.email_smtp import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.email_smtp import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.email_smtp import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
