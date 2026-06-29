# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p6.factories import create_grafana_oncall_notification_channel as _legacy_create_grafana_oncall_notification_channel

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.notification_channel.grafana_oncall.integration import (
    GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
    GrafanaOncallNotificationChannelIntegration,
    GrafanaOncallNotificationChannelIntegrationConfig,
    GrafanaOncallNotificationChannelClient,
)

__all__ = [
    "create_grafana_oncall_notification_channel",
    "create_grafana_oncall_notification_channel_integration",
]


def create_grafana_oncall_notification_channel_integration(
    *,
    client: GrafanaOncallNotificationChannelClient | None = None,
    enabled: bool = False,
) -> GrafanaOncallNotificationChannelIntegration:
    """
    Build a contract-based Grafana Oncall notification channel integration.

    The legacy facade (create_grafana_oncall_notification_channel) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Grafana Oncall notification channel integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GrafanaOncallNotificationChannelIntegration.from_client(client, enabled=enabled)
    return GrafanaOncallNotificationChannelIntegration.for_provider(
        provider_id=GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
        display_name="Grafana Oncall",
        config=GrafanaOncallNotificationChannelIntegrationConfig(enabled=enabled),
    )


def create_grafana_oncall_notification_channel(**kwargs: object) -> GrafanaOncallNotificationChannelIntegration:
    """Compatibility shim — constructs GrafanaOncallNotificationChannelIntegration from legacy runtime."""
    runtime = _legacy_create_grafana_oncall_notification_channel(**kwargs)
    if isinstance(runtime, GrafanaOncallNotificationChannelIntegration):
        return runtime
    return GrafanaOncallNotificationChannelIntegration.from_runtime(runtime)
