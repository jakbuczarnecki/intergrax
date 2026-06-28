# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID",
    "GrafanaOncallNotificationChannelIntegration",
    "GrafanaOncallNotificationChannelIntegrationConfig",
    "GrafanaOncallNotificationChannelClient",
    "create_grafana_oncall_notification_channel",
    "create_grafana_oncall_notification_channel_integration",
    "register_grafana_oncall_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_grafana_oncall_notification_channel",
        "create_grafana_oncall_notification_channel_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "GrafanaOncallNotificationChannelIntegration",
        "GrafanaOncallNotificationChannelIntegrationConfig",
        "GrafanaOncallNotificationChannelClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID",
        "GrafanaOncallNotificationChannelIntegration",
        "GrafanaOncallNotificationChannelIntegrationConfig",
        "GrafanaOncallNotificationChannelClient",
    }
)

def __getattr__(name: str):
    if name == "register_grafana_oncall_integration":
        from intergrax.integrations.providers.notification_channel.grafana_oncall.register import register_grafana_oncall_integration

        return register_grafana_oncall_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.notification_channel.grafana_oncall import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.grafana_oncall import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.notification_channel.grafana_oncall import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
