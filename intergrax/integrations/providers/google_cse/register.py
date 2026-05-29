# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Google CSE in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.google_cse.bundle import create_google_cse_search_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_google_cse_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.GOOGLE_CSE.value,
            categories=(IntegrationCategory.SEARCH_PROVIDER,),
            factory=create_google_cse_search_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_GOOGLE_CSE",
            description=(
                "Google Custom Search — web search via REST API "
                "(via create_google_cse_integration; legacy GOOGLE_CSE_* env supported)"
            ),
        ),
        override=override,
    )
