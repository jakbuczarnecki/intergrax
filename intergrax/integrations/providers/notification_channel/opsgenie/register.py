# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register opsgenie."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.opsgenie.bundle import create_opsgenie_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_opsgenie_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.OPSGENIE.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_opsgenie_notification_channel,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_OPSGENIE",
            description="opsgenie integration (Phase M.8 harness)",
        ),
        override=override,
    )
