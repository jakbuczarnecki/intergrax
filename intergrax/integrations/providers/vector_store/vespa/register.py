# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register vespa."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_vespa_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.VESPA.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_vespa_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_VESPA",
            description="vespa integration (Phase M.8 harness)",
        ),
        override=override,
    )
