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
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
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


def create_default_registry(
    embedding_model: Optional[str] = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingProviderRegistry:
    """
    TRANSITIONAL_RUNTIME_COMPATIBILITY — registry router backed by Integrations binding.

    Canonical provider construction authority is Integrations catalog + runtime binder.
    """
    _ensure_embedding_catalog_registered()

    registry = EmbeddingProviderRegistry()

    for slug in list_slugs(category=IntegrationCategory.EMBEDDING_PROVIDER):
        profile = EmbeddingProfile(provider=slug, model=embedding_model)

        def factory(
            resolved_slug: str = slug,
            resolved_profile: EmbeddingProfile = profile,
        ) -> EmbeddingProvider:
            return bind_embedding_provider(
                integration_profile=IntegrationProfile(embedding_provider=resolved_slug),
                embedding_profile=resolved_profile,
                execution_config=execution_config,
            )

        registry.register_factory(slug, factory)

    return registry


def create_default_embedding_manager() -> BaseEmbeddingManager:
    pipeline = create_default_embedding_pipeline()
    return EmbeddingManager(pipeline=pipeline)


def create_default_embedding_engine(
    registry: EmbeddingProviderRegistry | None = None,
    embedding_model: Optional[str] = None,
    *,
    provider: EmbeddingProvider | None = None,
) -> EmbeddingEngine:
    """Create EmbeddingEngine with Integrations-backed provider binding."""
    if provider is not None:
        return EmbeddingEngine(provider=provider)

    if registry is None:
        registry = create_default_registry(embedding_model=embedding_model)

    return EmbeddingEngine(registry=registry)


def create_default_embedding_pipeline(
    provider_id: Optional[str] = None,
    registry: EmbeddingProviderRegistry | None = None,
    embedding_model: Optional[str] = None,
    *,
    integration_profile: IntegrationProfile | None = None,
    embedding_profile: EmbeddingProfile | None = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingPipeline:
    """Create EmbeddingPipeline using canonical Integrations-backed runtime binding."""
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
    resolved_provider = resolved_profile.provider

    if integration_profile is None:
        integration_profile = IntegrationProfile(embedding_provider=resolved_provider)

    if registry is None:
        bound_provider = bind_embedding_provider(
            integration_profile=integration_profile,
            provider_id=provider_id,
            embedding_profile=resolved_profile,
            execution_config=execution_config,
        )
        engine = EmbeddingEngine(provider=bound_provider)
    else:
        engine = create_default_embedding_engine(
            registry=registry,
            embedding_model=resolved_model,
        )

    return EmbeddingPipeline(
        engine=engine,
        provider_id=resolved_provider,
    )
