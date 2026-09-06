# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine


pytestmark = pytest.mark.unit


def make_document(document_id: str, content: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant"},
            "content": content,
            "metadata": {"source": "test"},
            "provenance": {
                "source_kind": "test",
                "source_id": f"source-{document_id}",
            },
        }
    )


class FakeEmbeddingProvider(EmbeddingProvider):

    def provider_name(self) -> str:
        return "fake"

    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


def create_manager() -> BaseEmbeddingManager:

    provider = FakeEmbeddingProvider()
    engine = EmbeddingEngine(provider=provider)
    pipeline = EmbeddingPipeline(engine=engine)
    return EmbeddingManager(pipeline=pipeline)


def test_manager_embed_texts() -> None:

    manager = create_manager()

    result = manager.embed_texts(["a", "b"])

    assert result.shape == (2, 4)
    assert np.all(result == 1.0)


def test_manager_embed_one() -> None:

    manager = create_manager()

    result = manager.embed_one("hello")

    assert np.all(result == 1.0)


def test_manager_embed_documents() -> None:

    manager = create_manager()

    docs = [
        make_document("doc1", "doc1"),
        make_document("doc2", "doc2"),
    ]

    result = manager.embed_documents(docs)

    assert len(result.documents) == 2
    assert result.documents[0].content == "doc1"
    assert result.documents[1].content == "doc2"

    vector_0 = result.embeddings[0]
    vector_1 = result.embeddings[1]

    assert vector_0.shape == (4,)
    assert vector_1.shape == (4,)
    assert result.embeddings.shape == (2, 4)

    assert np.all(vector_0 == 1.0)
    assert np.all(vector_1 == 1.0)