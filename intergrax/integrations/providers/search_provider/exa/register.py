# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register exa."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.search_provider.exa.bundle import create_exa_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_exa_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.EXA.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_exa_search_provider,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_EXA",
            description="exa integration (Phase M.7)",
        ),
        override=override,
    )
