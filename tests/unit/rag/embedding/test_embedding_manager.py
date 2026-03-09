# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry
from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey


pytestmark = pytest.mark.unit


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


def create_manager() -> BaseEmbeddingManager:

    registry = EmbeddingProviderRegistry()
    registry.register(FakeEmbeddingProvider())

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id="fake",
    )

    manager = EmbeddingManager(pipeline=pipeline)

    return manager


def test_manager_embed_texts() -> None:

    manager = create_manager()

    result = manager.embed_texts(["a", "b"])

    assert result.shape == (2, 4)
    assert np.all(result == 1.0)


def test_manager_embed_one() -> None:

    manager = create_manager()

    result = manager.embed_one("hello")

    assert result.shape == (1, 4)
    assert np.all(result == 1.0)


def test_manager_embed_documents() -> None:

    manager = create_manager()

    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]

    result = manager.embed_documents(docs)

    assert len(result) == 2

    vector_0 = result[0].metadata[EmbeddingMetadataKey.VECTOR]
    vector_1 = result[1].metadata[EmbeddingMetadataKey.VECTOR]

    assert vector_0.shape == (4,)
    assert vector_1.shape == (4,)

    assert np.all(vector_0 == 1.0)
    assert np.all(vector_1 == 1.0)