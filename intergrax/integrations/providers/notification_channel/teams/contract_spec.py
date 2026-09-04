# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Teams notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.teams.bundle import (
    create_teams_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.teams.integration import (
    TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID,
    TeamsNotificationChannelIntegration,
    TeamsNotificationChannelIntegrationConfig,
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
    provider_id=TEAMS_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=TeamsNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_teams_notification_channel_integration,
    display_name="Teams",
    config_class=TeamsNotificationChannelIntegrationConfig,
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
