# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Pagerduty notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.pagerduty.bundle import (
    create_pagerduty_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.pagerduty.integration import (
    PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
    PagerdutyNotificationChannelIntegration,
    PagerdutyNotificationChannelIntegrationConfig,
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
    provider_id=PAGERDUTY_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=PagerdutyNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_pagerduty_notification_channel_integration,
    display_name="Pagerduty",
    config_class=PagerdutyNotificationChannelIntegrationConfig,
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
