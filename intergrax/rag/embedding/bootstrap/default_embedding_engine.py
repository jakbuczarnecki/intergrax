# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import os
from typing import Optional

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import (
    EmbeddingProviderRegistry,
    lazy_import_provider_factory,
)


def create_default_registry() -> EmbeddingProviderRegistry:
    registry = EmbeddingProviderRegistry()
    registry.register_factory(
        "hf",
        lazy_import_provider_factory(
            provider_id="hf",
            module_name="intergrax.rag.embedding.providers.hf_embedding_provider",
            class_name="HFEmbeddingProvider",
            dependency_name="sentence-transformers",
            extra_name="rag-local-embeddings",
        ),
    )
    registry.register_factory(
        "openai",
        lazy_import_provider_factory(
            provider_id="openai",
            module_name="intergrax.rag.embedding.providers.openai_embedding_provider",
            class_name="OpenAIEmbeddingProvider",
            dependency_name="openai",
        ),
    )
    registry.register_factory(
        "ollama",
        lazy_import_provider_factory(
            provider_id="ollama",
            module_name="intergrax.rag.embedding.providers.ollama_embedding_provider",
            class_name="OllamaEmbeddingProvider",
            dependency_name="langchain-ollama",
        ),
    )
    registry.register_factory(
        "vllm",
        lazy_import_provider_factory(
            provider_id="vllm",
            module_name="intergrax.rag.embedding.providers.vllm_embedding_provider",
            class_name="VllmEmbeddingProvider",
            dependency_name="openai",
        ),
    )
    registry.register_factory(
        "llama_cpp",
        lazy_import_provider_factory(
            provider_id="llama_cpp",
            module_name="intergrax.rag.embedding.providers.llama_cpp_embedding_provider",
            class_name="LlamaCppEmbeddingProvider",
            dependency_name="openai",
        ),
    )
    return registry


def create_default_embedding_manager()-> BaseEmbeddingManager:
    
    pipeline = create_default_embedding_pipeline()
    manager = EmbeddingManager(pipeline=pipeline)

    return manager


def create_default_embedding_engine(
    registry: EmbeddingProviderRegistry | None = None,
) -> EmbeddingEngine:
    """
    Create EmbeddingEngine with default embedding providers registered.

    Allows dependency override by providing a custom registry.
    """

    if registry is None:
        registry = create_default_registry()

    return EmbeddingEngine(
        registry=registry,
    )


def create_default_embedding_pipeline(
    provider_id: Optional[str] = None,
    registry: EmbeddingProviderRegistry | None = None,
) -> EmbeddingPipeline:
    """
    Create EmbeddingPipeline using the default embedding engine.
    """

    if registry is None:
        registry = create_default_registry()

    if provider_id is None:
        env_provider = os.getenv("INTERGRAX_RAG_EMBEDDING_PROVIDER", "").strip()
        provider_id = env_provider or registry.default_provider()

    engine = create_default_embedding_engine(registry=registry)

    return EmbeddingPipeline(
        engine=engine,
        provider_id=provider_id,
    )