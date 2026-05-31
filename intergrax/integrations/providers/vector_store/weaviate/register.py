# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register weaviate."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.vector_store.weaviate.bundle import create_weaviate_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_weaviate_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.WEAVIATE.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_weaviate_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_WEAVIATE",
            description="weaviate integration (Phase M.7)",
        ),
        override=override,
    )
