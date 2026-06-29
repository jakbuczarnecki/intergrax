# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "PagerdutyNotificationChannelIntegration",
    "PagerdutyNotificationChannelIntegrationConfig",
    "PagerdutyNotificationChannelClient",
    "create_pagerduty_integration",
    "create_pagerduty_notification_channel",
    "create_pagerduty_notification_channel_integration",
    "register_pagerduty_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_pagerduty_integration",
        "create_pagerduty_notification_channel",
        "create_pagerduty_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "PagerdutyNotificationChannelIntegration",
        "PagerdutyNotificationChannelIntegrationConfig",
        "PagerdutyNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "PagerdutyNotificationChannelIntegration",
        "PagerdutyNotificationChannelIntegrationConfig",
        "PagerdutyNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_pagerduty_integration":
        from intergrax.integrations.providers.notification_channel.pagerduty.register import register_pagerduty_integration

        return register_pagerduty_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.pagerduty import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.pagerduty import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.pagerduty import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
