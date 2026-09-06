# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from langchain_core.documents import Document

from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey


pytestmark = pytest.mark.integration


def create_manager() -> EmbeddingManager:

    provider = HFEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)
    pipeline = EmbeddingPipeline(engine=engine)
    return EmbeddingManager(pipeline=pipeline)


def test_hf_embedding_documents() -> None:

    manager = create_manager()

    docs = [
        Document(page_content="Artificial intelligence transforms software."),
        Document(page_content="Machine learning enables pattern discovery."),
    ]

    result = manager.embed_documents(docs)

    assert len(result.embeddings) == 2

    vector_0 = result.embeddings[0]
    vector_1 = result.embeddings[1]

    assert isinstance(vector_0, np.ndarray)
    assert isinstance(vector_1, np.ndarray)

    assert vector_0.ndim == 1
    assert vector_1.ndim == 1

    assert vector_0.shape == vector_1.shape

    assert not np.allclose(vector_0, vector_1)

    assert EmbeddingMetadataKey.EMBEDDING not in docs[0].metadata
    assert EmbeddingMetadataKey.EMBEDDING not in docs[1].metadata
