# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Mailpit notification channel."""

from __future__ import annotations

from intergrax.integrations.providers.notification_channel.mailpit.bundle import (
    create_mailpit_notification_channel_integration,
)
from intergrax.integrations.providers.notification_channel.mailpit.integration import (
    MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
    MailpitNotificationChannelIntegration,
    MailpitNotificationChannelIntegrationConfig,
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
    provider_id=MAILPIT_NOTIFICATION_CHANNEL_PROVIDER_ID,
    integration_class=MailpitNotificationChannelIntegration,
    contract_class=NotificationChannelIntegrationContract,
    contract_factory=create_mailpit_notification_channel_integration,
    display_name="Mailpit",
    config_class=MailpitNotificationChannelIntegrationConfig,
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
