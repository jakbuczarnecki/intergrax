# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register helicone."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.helicone.bundle import create_helicone_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_helicone_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.HELICONE.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_helicone_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_HELICONE",
            description="helicone integration (Phase M.8 harness)",
        ),
        override=override,
    )
