# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pagerduty."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.pagerduty.bundle import create_pagerduty_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_pagerduty_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PAGERDUTY.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_pagerduty_notification_channel,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_PAGERDUTY",
            description="pagerduty integration (Phase M.8 harness)",
        ),
        override=override,
    )
