# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrievalResult,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreHit

pytestmark = pytest.mark.unit


def _document() -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": "native content",
            "metadata": {"source": "test"},
            "provenance": {"source_kind": "test", "source_id": "doc-1"},
        }
    )


def test_retrieval_hit_revalidates_document_and_is_immutable() -> None:
    document = _document()
    hit = RetrievalHit(
        document=document,
        score=0.9,
        rank=0,
        channel="dense",
        vector_id="vector-1",
    )

    assert hit.document is not document
    assert hit.document.scope.tenant_id == "tenant-a"
    with pytest.raises(FrozenInstanceError):
        hit.rank = 1  # type: ignore[misc]


def test_retrieval_hit_validates_score_and_rank() -> None:
    with pytest.raises((TypeError, ValueError)):
        RetrievalHit(document=_document(), score=float("nan"), rank=0, channel="dense")
    with pytest.raises((TypeError, ValueError)):
        RetrievalHit(document=_document(), score=True, rank=0, channel="dense")  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        RetrievalHit(document=_document(), score=0.5, rank=True, channel="dense")  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        RetrievalHit(document=_document(), score=0.5, rank=-1, channel="dense")


def test_retrieval_hit_embedding_is_float32_readonly_and_defensively_copied() -> None:
    embedding = np.array([1.0, 2.0], dtype=np.float64)
    hit = RetrievalHit(
        document=_document(),
        score=0.9,
        rank=0,
        channel="dense",
        embedding=embedding,
    )

    assert hit.embedding is not None
    assert hit.embedding.dtype == np.float32
    assert hit.embedding.flags.writeable is False
    embedding[0] = 99.0
    assert hit.embedding[0] == 1.0
    with pytest.raises(ValueError):
        hit.embedding[0] = 3.0


def test_retrieval_hit_maps_native_provider_result_without_reconstructing_shape() -> None:
    provider_hit = VectorStoreHit(
        vector_id="vector-1",
        document=_document(),
        similarity_score=0.9,
        rank=4,
        embedding=[1.0, 2.0],
    )

    hit = RetrievalHit.from_vector_store_hit(
        provider_hit,
        channel="dense",
        retriever_name="vector_similarity",
    )

    assert hit.document.identity.document_id == "doc-1"
    assert hit.vector_id == "vector-1"
    assert hit.source_rank == 4
    assert hit.rank == 4


def test_retrieval_result_serialization_is_stable() -> None:
    hit = RetrievalHit(
        document=_document(),
        score=0.9,
        rank=0,
        channel="dense",
        vector_id="vector-1",
    )
    result = RetrievalResult(
        hits=(hit,),
        query="test",
        requested_top_k=1,
        retrieval_mode="vector_similarity",
    )

    assert result.hits == (hit,)
    assert result.to_json() == result.to_json()
    assert result.to_dict()["hits"][0]["document"]["scope"]["tenant_id"] == "tenant-a"
