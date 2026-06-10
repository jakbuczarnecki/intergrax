# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import List, Sequence

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.bootstrap.hierarchical_bootstrap import profile_uses_hierarchical_index
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import create_default_embedding_manager
from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.hierarchical_retriever import HierarchicalRetriever
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _InlineLoader:
    def __init__(self, docs: Sequence[Document]) -> None:
        self._docs = list(docs)

    def load_document(self, source: str, **kwargs: object) -> List[Document]:
        del source, kwargs
        return list(self._docs)


class _InlineSplitter:
    def __init__(self, strategy: ParentChildChunkingStrategy) -> None:
        self._strategy = strategy

    def split_documents(self, docs: Sequence[Document], strategy_id: str | None = None) -> Sequence[Document]:
        del strategy_id
        return self._strategy.chunk(docs)


def test_profile_uses_hierarchical_index_flag_and_retriever() -> None:
    assert profile_uses_hierarchical_index(RagProfile(hierarchical_index_enabled=True)) is True
    assert profile_uses_hierarchical_index(RagProfile(retriever_id="hierarchical")) is True
    assert profile_uses_hierarchical_index(RagProfile()) is False


def test_rag_stack_wires_toc_store_when_hierarchical_enabled() -> None:
    profile = RagProfile(
        hierarchical_index_enabled=True,
        retriever_id="hierarchical",
        enable_rerank=False,
        route_mode="off",
    )
    stack = create_default_rag_stack(profile=profile)
    assert stack.toc_vectorstore_manager is not None
    assert stack.toc_vectorstore_manager is not stack.vectorstore_manager


def test_dual_index_ingest_and_hierarchical_retrieve_expands_parents(tmp_path) -> None:
    source = tmp_path / "book.txt"
    book_text = "alpha " * 200 + "beta " * 200 + "gamma " * 200
    source.write_text(book_text, encoding="utf-8")

    chunks_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    toc_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    embedding = create_default_embedding_manager()
    strategy = ParentChildChunkingStrategy(parent_size=400, child_size=80, child_overlap=10)

    pipeline = IngestPipeline(
        loader=_InlineLoader([Document(page_content=book_text, metadata={"doc_id": "book-1"})]),
        splitter=_InlineSplitter(strategy),
        embedding_manager=embedding,
        vectorstore=chunks_store,
        toc_vectorstore=toc_store,
        profile=RagProfile(
            hierarchical_index_enabled=True,
            chunking_strategy_id="parent_child",
            enable_rerank=False,
        ),
    )
    ingest = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "book"},
            chunking_strategy_id="parent_child",
        )
    )
    assert ingest.used is True
    assert ingest.num_chunks > 0
    assert chunks_store.count() > 0
    assert toc_store.count() > 0

    retriever_manager = create_default_retriever_manager(
        vector_store=chunks_store,
        embedding_manager=embedding,
        toc_vector_store=toc_store,
    )
    hierarchical = retriever_manager._pipeline._engine.get_retriever("hierarchical")  # type: ignore[attr-defined]
    assert isinstance(hierarchical, HierarchicalRetriever)
    assert hierarchical._toc is toc_store

    service = RetrievalService(
        retriever_manager=retriever_manager,
        profile=RagProfile(retriever_id="hierarchical", enable_rerank=False, route_mode="off"),
    )
    result = service.retrieve(RetrievalRequest(query="alpha beta content", top_k=5))
    assert result.used is True
    assert result.trace.retriever_id == "hierarchical"
    assert len(result.chunks) > 0
