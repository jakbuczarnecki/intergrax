# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``cohere_rerank`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="cohere_rerank",
    categories=(IntegrationCategory.RERANK_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_COHERE_RERANK',
    description='Cohere rerank API for RAG retrieval ordering',
)
