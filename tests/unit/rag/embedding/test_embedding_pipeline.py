# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
import numpy as np

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.pipeline.embedding_pipeline import EmbeddingPipeline
from intergrax.rag.embedding.engine.embedding_engine import EmbeddingEngine
from intergrax.rag.embedding.registry.embedding_provider_registry import EmbeddingProviderRegistry


pytestmark = pytest.mark.unit


def make_document(
    document_id: str,
    content: str = "document",
    *,
    metadata: dict[str, object] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant"},
            "content": content,
            "metadata": metadata or {"source": "test"},
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
        make_document("doc1", "doc1"),
        make_document("doc2", "doc2"),
    ]

    result = pipeline.embed_documents(docs)

    assert result.documents == tuple(docs)
    assert result.embeddings.shape == (2, 5)
    assert np.all(result.embeddings == 1.0)
    assert all("embedding" not in document.metadata for document in result.documents)


def test_embedding_result_normalizes_and_defensively_copies_embeddings() -> None:
    documents = [make_document("one"), make_document("two")]
    source = np.array([[1, 2], [3, 4]], dtype=np.float64)

    result = EmbeddingResult(documents=documents, embeddings=source)
    source[0, 0] = 99

    assert result.documents == tuple(documents)
    assert result.embeddings.dtype == np.float32
    assert result.embeddings.shape == (2, 2)
    assert np.array_equal(result.embeddings, [[1, 2], [3, 4]])
    assert result.embeddings.flags.writeable is False


@pytest.mark.parametrize(
    "embeddings",
    [
        np.ones(2, dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float32),
    ],
)
def test_embedding_result_rejects_wrong_dimensions(embeddings: np.ndarray) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        EmbeddingResult(documents=(make_document("one"),), embeddings=embeddings)


def test_embedding_result_rejects_wrong_cardinality() -> None:
    with pytest.raises(ValueError, match="document count"):
        EmbeddingResult(
            documents=(make_document("one"), make_document("two")),
            embeddings=np.ones((1, 2), dtype=np.float32),
        )


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_embedding_result_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        EmbeddingResult(
            documents=(make_document("one"),),
            embeddings=np.array([[value]], dtype=np.float32),
        )


def test_embedding_result_rejects_foreign_document_type() -> None:
    with pytest.raises(TypeError, match="KnowledgeDocument"):
        EmbeddingResult(
            documents=("not-a-document",),  # type: ignore[arg-type]
            embeddings=np.ones((1, 1), dtype=np.float32),
        )


def test_embedding_result_accepts_empty_result() -> None:
    result = EmbeddingResult(
        documents=(),
        embeddings=np.empty((0, 0), dtype=np.float32),
    )

    assert result.documents == ()
    assert result.embeddings.shape == (0, 0)
    assert result.embeddings.flags.writeable is False


def test_embedding_result_rejects_nonempty_empty_matrix() -> None:
    with pytest.raises(ValueError, match="document count"):
        EmbeddingResult(
            documents=(make_document("one"),),
            embeddings=np.empty((0, 0), dtype=np.float32),
        )


def test_pipeline_validates_documents_and_calls_engine_once() -> None:
    class SpyEngine:
        calls = 0
        received_texts: list[str] = []

        def embed(self, texts: list[str], *, provider_id: str) -> np.ndarray:
            self.calls += 1
            self.received_texts = list(texts)
            assert provider_id == "spy"
            return np.ones((len(texts), 2), dtype=np.float32)

    engine = SpyEngine()
    pipeline = EmbeddingPipeline(engine=engine, provider_id="spy")
    documents = [make_document("one", "first"), make_document("two", "second")]
    original_documents = [document.model_dump(mode="python") for document in documents]

    result = pipeline.embed_documents(iter(documents))

    assert engine.calls == 1
    assert engine.received_texts == ["first", "second"]
    assert result.documents == tuple(documents)
    assert [document.model_dump(mode="python") for document in documents] == original_documents


def test_pipeline_rejects_foreign_document_type() -> None:
    pipeline = create_pipeline()

    with pytest.raises(TypeError, match="KnowledgeDocument"):
        pipeline.embed_documents(["not-a-document"])  # type: ignore[arg-type]


def test_pipeline_empty_input_returns_empty_result_without_engine_call() -> None:
    class FailingEngine:
        def embed(self, texts: list[str], *, provider_id: str) -> np.ndarray:
            raise AssertionError("empty input must not call the engine")

    pipeline = EmbeddingPipeline(engine=FailingEngine(), provider_id="unused")

    result = pipeline.embed_documents([])

    assert result.documents == ()
    assert result.embeddings.shape == (0, 0)


def test_pipeline_propagates_engine_errors() -> None:
    class FailingEngine:
        def embed(self, texts: list[str], *, provider_id: str) -> np.ndarray:
            raise RuntimeError("provider failed")

    pipeline = EmbeddingPipeline(engine=FailingEngine(), provider_id="failing")

    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.embed_documents([make_document("one")])


def test_pipeline_rejects_engine_cardinality_mismatch() -> None:
    class WrongCardinalityEngine:
        def embed(self, texts: list[str], *, provider_id: str) -> np.ndarray:
            return np.ones((1, 2), dtype=np.float32)

    pipeline = EmbeddingPipeline(engine=WrongCardinalityEngine(), provider_id="wrong")

    with pytest.raises(ValueError, match="document count"):
        pipeline.embed_documents([make_document("one"), make_document("two")])


def test_embedding_dimension_matches_provider() -> None:

    pipeline = create_pipeline()

    result = pipeline.embed_texts(["a", "b"])

    dimension = result.shape[1]

    assert dimension > 0

    for row in result:
        assert row.shape[0] == dimension