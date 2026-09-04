# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Opsgenie notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.opsgenie.bundle import (
    create_opsgenie_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.opsgenie.integration import (
    OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
    OpsgenieNotificationChannelIntegration,
    OpsgenieNotificationChannelIntegrationConfig,
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
    provider_id=OPSGENIE_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=OpsgenieNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_opsgenie_notification_channel_integration,
    display_name="Opsgenie",
    config_class=OpsgenieNotificationChannelIntegrationConfig,
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
