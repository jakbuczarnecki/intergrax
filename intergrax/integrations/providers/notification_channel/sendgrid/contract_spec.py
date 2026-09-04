# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Sendgrid notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.sendgrid.bundle import (
    create_sendgrid_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.sendgrid.integration import (
    SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID,
    SendgridNotificationChannelIntegration,
    SendgridNotificationChannelIntegrationConfig,
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
    provider_id=SENDGRID_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=SendgridNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_sendgrid_notification_channel_integration,
    display_name="Sendgrid",
    config_class=SendgridNotificationChannelIntegrationConfig,
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
