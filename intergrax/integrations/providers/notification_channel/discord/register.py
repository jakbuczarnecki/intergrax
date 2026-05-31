# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register discord."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.discord.bundle import create_discord_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_discord_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.DISCORD.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_discord_notification_channel,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_DISCORD",
            description="discord integration (Phase M.7)",
        ),
        override=override,
    )
