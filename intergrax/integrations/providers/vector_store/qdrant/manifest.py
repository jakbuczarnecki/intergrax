# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``qdrant`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="qdrant",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_QDRANT',
    description='Qdrant vector store catalog bridge — delegates to intergrax/rag/ QdrantVectorStore',
)
