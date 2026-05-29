# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register lab JSON in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.lab_json.bundle import create_lab_json_interaction_surface
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_lab_json_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.LAB_JSON.value,
            categories=(IntegrationCategory.INTERACTION_SURFACE,),
            factory=create_lab_json_interaction_surface,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_LAB_JSON",
            description=(
                "Laboratory JSON interaction surface — vendor-neutral dict → Task "
                "(via create_lab_json_integration)"
            ),
        ),
        override=override,
    )
