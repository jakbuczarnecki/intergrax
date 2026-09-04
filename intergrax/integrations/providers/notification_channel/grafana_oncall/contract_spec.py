# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Grafana Oncall notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.grafana_oncall.bundle import (
    create_grafana_oncall_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.grafana_oncall.integration import (
    GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
    GrafanaOncallNotificationChannelIntegration,
    GrafanaOncallNotificationChannelIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.messaging import (
    NotificationChannelIntegrationContract,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="notification_channel",
    provider_id=GRAFANA_ONCALL_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=GrafanaOncallNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_grafana_oncall_notification_channel_integration,
    display_name="Grafana Oncall",
    config_class=GrafanaOncallNotificationChannelIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={
        "source": "explicit_provider_declaration"
    },
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
