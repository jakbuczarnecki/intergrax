# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.embedding_metadata_key import EmbeddingMetadataKey
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry


pytestmark = pytest.mark.unit


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 5

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 5), dtype=np.float32)


def create_pipeline() -> EmbeddingPipeline:

    registry = EmbeddingProviderRegistry()

    provider = FakeEmbeddingProvider()

    registry.register(provider)

    engine = EmbeddingEngine(registry)

    pipeline = EmbeddingPipeline(
        engine=engine,
        provider_id="fake",
    )

    return pipeline


def test_embed_texts() -> None:

    pipeline = create_pipeline()

    result = pipeline.embed_texts(["a", "b", "c"])

    assert result.shape == (3, 5)
    assert np.all(result == 1.0)


def test_embed_one() -> None:

    pipeline = create_pipeline()

    result = pipeline.embed_one("hello")

    assert result.shape == (1, 5)
    assert np.all(result == 1.0)


def test_embed_documents() -> None:

    pipeline = create_pipeline()

    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
    ]

    result = pipeline.embed_documents(docs)

    assert len(result) == 2

    assert result[0].page_content == "doc1"
    assert result[1].page_content == "doc2"

    vector_0 = result[0].metadata[EmbeddingMetadataKey.VECTOR]
    vector_1 = result[1].metadata[EmbeddingMetadataKey.VECTOR]

    assert vector_0.shape == (5,)
    assert vector_1.shape == (5,)

    assert np.all(vector_0 == 1.0)
    assert np.all(vector_1 == 1.0)


def test_embedding_dimension_matches_provider() -> None:

    pipeline = create_pipeline()

    result = pipeline.embed_texts(["a", "b"])

    dimension = result.shape[1]

    assert dimension > 0

    for row in result:
        assert row.shape[0] == dimension