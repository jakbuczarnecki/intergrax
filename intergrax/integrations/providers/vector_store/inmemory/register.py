# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register inmemory."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_inmemory_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.INMEMORY.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_inmemory_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_INMEMORY",
            description="inmemory integration (Phase M.7)",
        ),
        override=override,
    )
