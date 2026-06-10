# © Artur Czarnecki. All rights reserved.

"""M-RAG.29 — formal Citation on engine + catalog rag.retrieve output."""

from __future__ import annotations

from typing import List, Optional, Sequence

import pytest

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.citation import citation_from_chunk, citations_from_chunks
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.retrievers.contracts.base_retriever_manager import BaseRetrieverManager
from intergrax.rag.vectorstore.contracts.vector_store import VectorStoreHit
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.gate


class FakeEmbeddingManager:
    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorstoreManager:
    def __init__(self, hits: List[VectorStoreHit]) -> None:
        self._hits = hits

    def query(
        self,
        *,
        query_embedding,
        top_k: int,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:
        return list(self._hits)


class StubRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "stub"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        return [
            RetrieverCandidate(
                id="chunk-1",
                content="Contract clause 4.2 applies.",
                metadata={
                    "doc_id": "contract-2024",
                    "source": "contract.pdf",
                    "page": 12,
                    "url": "https://example.test/contract.pdf",
                },
                score=0.92,
            )
        ]


class StubRetrieverManager(BaseRetrieverManager):
    def retrieve(
        self,
        query_text: str,
        *,
        retriever_id: str,
        query_embedding: Sequence[float] | None = None,
        top_k: int = 5,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> List[RetrieverCandidate]:
        return StubRetriever().retrieve(
            RetrieverQuery(
                query_text=query_text,
                query_embedding=None,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        )

    def retrieve_query(self, query: RetrieverQuery, retriever_id: str) -> List[RetrieverCandidate]:
        return self.retrieve(query.query_text, retriever_id=retriever_id, top_k=query.top_k)


def test_citation_from_chunk_preserves_source_metadata() -> None:
    chunk = RetrievalChunk(
        id="chunk-1",
        text="Contract clause 4.2 applies.",
        score=0.92,
        metadata={
            "doc_id": "contract-2024",
            "source": "contract.pdf",
            "page": 12,
            "url": "https://example.test/contract.pdf",
        },
    )
    citation = citation_from_chunk(chunk)

    assert citation.chunk_id == "chunk-1"
    assert citation.source_id == "contract-2024"
    assert citation.source_label == "contract.pdf"
    assert citation.page == 12
    assert citation.url == "https://example.test/contract.pdf"
    assert citation.score == 0.92
    assert "clause 4.2" in (citation.excerpt or "")


def test_retrieval_service_emits_formal_citations() -> None:
    service = RetrievalService(
        retriever_manager=StubRetrieverManager(),
        profile=RagProfile(enable_rerank=False),
    )
    result = service.retrieve(RetrievalRequest(query="clause 4.2"))

    assert result.used is True
    assert len(result.citations) == 1
    assert result.citations[0].source_id == "contract-2024"
    assert citations_from_chunks(result.chunks)[0].source_id == result.citations[0].source_id


def test_rag_retrieve_output_preserves_engine_citations() -> None:
    hits = [
        VectorStoreHit(
            id="chunk-1",
            content="Intergrax is an agent runtime.",
            metadata={
                "doc_id": "readme-1",
                "source": "readme.md",
                "page_number": 3,
            },
            similarity_score=0.91,
            rank=1,
        )
    ]
    ctx = ToolWiringContext(
        vectorstore_manager=FakeVectorstoreManager(hits),
        embedding_manager=FakeEmbeddingManager(),
        rag_profile=RagProfile(enable_rerank=False, route_mode="off", retriever_id="vector_similarity"),
    )

    out = perform_rag_retrieve(ctx, RagRetrieveInput(query="What is Intergrax?", top_k=3))

    assert out.used is True
    assert len(out.citations) == 1
    assert out.citations[0].source_id == "readme-1"
    assert out.citations[0].source_label == "readme.md"
    assert out.citations[0].page == 3
    assert out.diagnostics.get("citation_count") == 1
