# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register slash_command interaction surface."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.interaction_surface.slash_command.bundle import create_slash_command_interaction_surface
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_slash_command_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SLASH_COMMAND.value,
            categories=(IntegrationCategory.INTERACTION_SURFACE,),
            factory=create_slash_command_interaction_surface,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SLASH_COMMAND",
            description="Generic slash-command intake (Slack/Teams/CLI payloads)",
        ),
        override=override,
    )
