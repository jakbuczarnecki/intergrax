# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.rerank_provider.cohere_rerank.bundle import create_cohere_rerank_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_cohere_rerank_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.COHERE_RERANK.value,
            categories=(IntegrationCategory.RERANK_PROVIDER,),
            factory=create_cohere_rerank_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_COHERE_RERANK",
            description="Cohere rerank API for RAG retrieval ordering",
        ),
        override=override,
    )
