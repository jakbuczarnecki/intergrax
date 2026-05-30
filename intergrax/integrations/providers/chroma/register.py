# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Chroma in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_chroma_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.CHROMA.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_chroma_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_CHROMA",
            description="Chroma vector store catalog bridge — delegates to intergrax/rag/ ChromaVectorStore",
        ),
        override=override,
    )
