# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Teams in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.teams.bundle import create_teams_catalog_factory
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_teams_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.TEAMS.value,
            categories=(
                IntegrationCategory.NOTIFICATION_CHANNEL,
                IntegrationCategory.INTERACTION_SURFACE,
            ),
            factory=create_teams_catalog_factory,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_TEAMS",
            description=(
                "Microsoft Teams — outbound webhook notifications + inbound activities "
                "(via create_teams_integration)"
            ),
        ),
        override=override,
    )
