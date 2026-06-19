# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest
from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.providers.vllm_embedding_provider import VllmEmbeddingProvider
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from testing_support.builder import require_vllm_embed_reachable

pytestmark = pytest.mark.integration


def test_vllm_embedding_documents() -> None:
    require_vllm_embed_reachable()

    registry = EmbeddingProviderRegistry()
    provider = VllmEmbeddingProvider()
    registry.register(provider)

    engine = EmbeddingEngine(registry)
    pipeline = EmbeddingPipeline(engine=engine, provider_id=provider.provider_name())
    manager = EmbeddingManager(pipeline=pipeline)

    docs = [
        Document(page_content="Embeddings enable semantic search."),
        Document(page_content="Vector similarity powers RAG systems."),
    ]

    result = manager.embed_documents(docs)

    assert len(result.embeddings) == 2
    vector_0 = result.embeddings[0]
    vector_1 = result.embeddings[1]

    assert isinstance(vector_0, np.ndarray)
    assert isinstance(vector_1, np.ndarray)
    assert vector_0.ndim == 1
    assert vector_1.ndim == 1
    assert vector_0.shape[0] > 0
    assert vector_0.shape[0] == vector_1.shape[0]
    assert vector_0.shape[0] == provider.dimension()

    assert result.documents[0].page_content == docs[0].page_content
    assert result.documents[1].page_content == docs[1].page_content
