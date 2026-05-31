# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register milvus."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_milvus_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.MILVUS.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_milvus_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_MILVUS",
            description="milvus integration (Phase M.7)",
        ),
        override=override,
    )
