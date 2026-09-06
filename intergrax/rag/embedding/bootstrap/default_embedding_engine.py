# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.catalog import list_slugs
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.profile import EmbeddingProfile, embedding_profile_from_env
from intergrax.rag.embedding.runtime.resolver import (
    bind_embedding_provider,
    resolve_embedding_provider_slug,
)


def _ensure_embedding_catalog_registered() -> None:
    if list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER):
        return
    from intergrax.integrations.registry.bootstrap import register_default_integrations

    register_default_integrations(preset="full")


def create_default_embedding_manager() -> BaseEmbeddingManager:
    pipeline = create_default_embedding_pipeline()
    return EmbeddingManager(pipeline=pipeline)


def create_default_embedding_engine(
    provider_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    *,
    provider: EmbeddingProvider | None = None,
    integration_profile: IntegrationProfile | None = None,
    embedding_profile: EmbeddingProfile | None = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingEngine:
    """Create EmbeddingEngine with Integrations-backed provider binding."""
    if provider is not None:
        return EmbeddingEngine(provider=provider)

    _ensure_embedding_catalog_registered()

    compat_profile = embedding_profile or embedding_profile_from_env()
    resolved_model = embedding_model if embedding_model is not None else compat_profile.model
    resolved_profile = EmbeddingProfile(
        provider=resolve_embedding_provider_slug(
            integration_profile=integration_profile,
            provider_id=provider_id,
            embedding_profile=compat_profile,
        ),
        model=resolved_model,
    )

    if integration_profile is None:
        integration_profile = IntegrationProfile(embedding_provider=resolved_profile.provider)

    bound_provider = bind_embedding_provider(
        integration_profile=integration_profile,
        provider_id=provider_id,
        embedding_profile=resolved_profile,
        execution_config=execution_config,
    )
    return EmbeddingEngine(provider=bound_provider)


def create_default_embedding_pipeline(
    provider_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    *,
    integration_profile: IntegrationProfile | None = None,
    embedding_profile: EmbeddingProfile | None = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingPipeline:
    """Create EmbeddingPipeline using canonical Integrations-backed runtime binding."""
    engine = create_default_embedding_engine(
        provider_id=provider_id,
        embedding_model=embedding_model,
        integration_profile=integration_profile,
        embedding_profile=embedding_profile,
        execution_config=execution_config,
    )
    return EmbeddingPipeline(engine=engine)
