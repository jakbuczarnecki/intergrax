# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register generic webhook in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.webhook.bundle import create_webhook_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_webhook_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.WEBHOOK.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_webhook_notification_channel,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_WEBHOOK",
            description=(
                "Generic HTTP webhook notification channel "
                "(via create_webhook_integration / GenericJsonPayloadFormatter)"
            ),
        ),
        override=override,
    )
