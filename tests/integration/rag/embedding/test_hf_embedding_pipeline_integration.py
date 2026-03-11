# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey


pytestmark = pytest.mark.integration


def create_manager() -> EmbeddingManager:

    registry = EmbeddingProviderRegistry()

    provider = HFEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id=provider.provider_name(),
    )

    manager = EmbeddingManager(pipeline=pipeline)

    return manager


def test_hf_embedding_documents() -> None:

    manager = create_manager()

    docs = [
        Document(page_content="Artificial intelligence transforms software."),
        Document(page_content="Machine learning enables pattern discovery."),
    ]

    result = manager.embed_documents(docs)

    assert len(result) == 2

    vector_0 = result[0].metadata[EmbeddingMetadataKey.VECTOR]
    vector_1 = result[1].metadata[EmbeddingMetadataKey.VECTOR]

    assert isinstance(vector_0, np.ndarray)
    assert isinstance(vector_1, np.ndarray)

    assert vector_0.ndim == 1
    assert vector_1.ndim == 1

    assert vector_0.shape[0] > 0
    assert vector_1.shape[0] > 0

    assert result[0].page_content == docs[0].page_content
    assert result[1].page_content == docs[1].page_content


def test_hf_embedding_batch_texts() -> None:

    manager = create_manager()

    texts = [
        "Vector databases store embeddings.",
        "Embeddings represent semantic meaning.",
        "Retrieval augmented generation uses embeddings.",
    ]

    result = manager.embed_texts(texts)

    assert result.shape[0] == 3
    assert result.shape[1] > 0