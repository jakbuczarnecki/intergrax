# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderRegistry,
)
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.profile import embedding_profile_from_env
from intergrax.rag.embedding.registry.provider_factory_registration import (
    build_hf_provider_factory,
    build_llama_cpp_provider_factory,
    build_ollama_provider_factory,
    build_openai_provider_factory,
    build_vllm_provider_factory,
)


def create_default_registry(
    embedding_model: Optional[str] = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingProviderRegistry:
    registry = EmbeddingProviderRegistry()
    registry.register_factory(
        "hf",
        build_hf_provider_factory(
            embedding_model=embedding_model,
            execution_config=execution_config,
        ),
    )
    registry.register_factory(
        "openai",
        build_openai_provider_factory(embedding_model=embedding_model),
    )
    registry.register_factory(
        "ollama",
        build_ollama_provider_factory(embedding_model=embedding_model),
    )
    registry.register_factory(
        "vllm",
        build_vllm_provider_factory(embedding_model=embedding_model),
    )
    registry.register_factory(
        "llama_cpp",
        build_llama_cpp_provider_factory(embedding_model=embedding_model),
    )
    return registry


def create_default_embedding_manager()-> BaseEmbeddingManager:
    
    pipeline = create_default_embedding_pipeline()
    manager = EmbeddingManager(pipeline=pipeline)

    return manager


def create_default_embedding_engine(
    registry: EmbeddingProviderRegistry | None = None,
    embedding_model: Optional[str] = None,
) -> EmbeddingEngine:
    """
    Create EmbeddingEngine with default embedding providers registered.

    Allows dependency override by providing a custom registry.
    """

    if registry is None:
        registry = create_default_registry(embedding_model=embedding_model)

    return EmbeddingEngine(
        registry=registry,
    )


def create_default_embedding_pipeline(
    provider_id: Optional[str] = None,
    registry: EmbeddingProviderRegistry | None = None,
    embedding_model: Optional[str] = None,
) -> EmbeddingPipeline:
    """
    Create EmbeddingPipeline using the default embedding engine.
    """

    profile = embedding_profile_from_env()
    resolved_model = embedding_model if embedding_model is not None else profile.model

    if registry is None:
        registry = create_default_registry(embedding_model=resolved_model)

    resolved_provider = provider_id or profile.provider

    engine = create_default_embedding_engine(
        registry=registry,
        embedding_model=resolved_model,
    )

    return EmbeddingPipeline(
        engine=engine,
        provider_id=resolved_provider,
    )
