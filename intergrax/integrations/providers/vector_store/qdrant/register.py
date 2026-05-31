# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Qdrant in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_qdrant_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.QDRANT.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_qdrant_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_QDRANT",
            description="Qdrant vector store catalog bridge — delegates to intergrax/rag/ QdrantVectorStore",
        ),
        override=override,
    )
