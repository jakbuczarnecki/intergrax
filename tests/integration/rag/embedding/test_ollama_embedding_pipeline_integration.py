# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np
import requests

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.embedding.providers.ollama_embedding_provider import OllamaEmbeddingProvider

pytestmark = pytest.mark.integration


def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not ollama_available(), reason="Ollama server not available")
def test_ollama_embedding_documents() -> None:

    registry = EmbeddingProviderRegistry()

    provider = OllamaEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id=provider.provider_name(),
    )

    manager = EmbeddingManager(pipeline=pipeline)

    docs = [
        Document(page_content="Embeddings enable semantic search."),
        Document(page_content="Vector similarity powers RAG systems."),
    ]

    result = manager.embed_documents(docs)

    assert len(result) == 2

    vector_0 = result.embeddings[0]
    vector_1 = result.embeddings[1]

    assert isinstance(vector_0, np.ndarray)
    assert isinstance(vector_1, np.ndarray)

    assert vector_0.ndim == 1
    assert vector_1.ndim == 1

    assert vector_0.shape[0] > 0
    assert vector_1.shape[0] > 0

    assert result.documents[0].page_content == docs[0].page_content
    assert result.documents[1].page_content == docs[1].page_content