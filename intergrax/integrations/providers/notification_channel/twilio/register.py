# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register twilio."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.twilio.bundle import create_twilio_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_twilio_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.TWILIO.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_twilio_notification_channel,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_TWILIO",
            description="twilio integration (Phase M.7)",
        ),
        override=override,
    )
