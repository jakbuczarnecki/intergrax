# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Pinecone in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.pinecone.bundle import create_pinecone_vector_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_pinecone_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PINECONE.value,
            categories=(IntegrationCategory.VECTOR_STORE,),
            factory=create_pinecone_vector_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_PINECONE",
            description=(
                "Pinecone vector store catalog bridge — delegates to intergrax/rag/ PineconeVectorStore"
            ),
        ),
        override=override,
    )
