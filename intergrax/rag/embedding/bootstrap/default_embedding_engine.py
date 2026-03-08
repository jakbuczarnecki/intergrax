# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.rag.embedding.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider
from intergrax.rag.embedding.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from intergrax.rag.embedding.providers.openai_embedding_provider import OpenAIEmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry



def create_default_embedding_engine(
    registry: EmbeddingProviderRegistry | None = None,
) -> EmbeddingEngine:
    """
    Create EmbeddingEngine with default embedding providers registered.

    Allows dependency override by providing a custom registry.
    """

    if registry is None:
        registry = EmbeddingProviderRegistry()

        registry.register(HFEmbeddingProvider())
        registry.register(OpenAIEmbeddingProvider())
        registry.register(OllamaEmbeddingProvider())

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

    if provider_id is None:
        provider_id = registry.default_provider()

    engine = create_default_embedding_engine(registry=registry)

    return EmbeddingPipeline(
        engine=engine,
        provider_id=provider_id,
    )