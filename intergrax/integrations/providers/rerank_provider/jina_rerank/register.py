# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.rerank_provider.jina_rerank.bundle import create_jina_rerank_provider
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_jina_rerank_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.JINA_RERANK.value,
            categories=(IntegrationCategory.RERANK_PROVIDER,),
            factory=create_jina_rerank_provider,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_JINA_RERANK",
            description="Jina rerank API for RAG retrieval ordering",
        ),
        override=override,
    )
