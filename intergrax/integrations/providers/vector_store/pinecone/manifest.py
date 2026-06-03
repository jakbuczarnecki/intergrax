# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``pinecone`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="pinecone",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_PINECONE',
    description='Pinecone vector store catalog bridge — delegates to intergrax/rag/ PineconeVectorStore',
)
