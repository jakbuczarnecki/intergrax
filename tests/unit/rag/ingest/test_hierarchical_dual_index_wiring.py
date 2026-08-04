# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.bootstrap.hierarchical_bootstrap import profile_uses_hierarchical_index
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from langchain_core.documents import Document
from testing_support.builder import build_fake_embedding_manager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _InlineLoader:
    def __init__(self, docs: Sequence[KnowledgeDocument]) -> None:
        self._docs = list(docs)

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del source, kwargs
        return list(self._docs)


class _InlineSplitter:
    def __init__(self, strategy: ParentChildChunkingStrategy) -> None:
        self._strategy = strategy

    def split_documents(self, docs: Sequence[KnowledgeDocument], strategy_id: str | None = None) -> Sequence[KnowledgeDocument]:
        del strategy_id
        return self._strategy.chunk(docs)


def test_dual_index_strategy_uses_embed_texts_for_main_and_toc() -> None:
    class _EmbeddingSpy:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_documents(self, documents: object) -> None:
            raise AssertionError("dual indexing compatibility must use embed_texts")

        def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
            self.calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_documents(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    documents = [
        Document(
            page_content="chunk-a",
            metadata={ChunkMetadataKey.SECTION: "Section A"},
        ),
        Document(
            page_content="chunk-b",
            metadata={ChunkMetadataKey.SECTION: "Section B"},
        ),
    ]
    embedding = _EmbeddingSpy()
    main_store = _Store()
    toc_store = _Store()

    DualIndexStrategy(toc_vectorstore=toc_store, batch_size=10).build_index(
        documents=documents,
        embed_manager=embedding,
        vectorstore=main_store,
    )

    assert embedding.calls == [["chunk-a", "chunk-b"], ["Section A", "Section B"]]
    assert main_store.calls[0]["documents"] == documents
    assert toc_store.calls[0]["documents"][0].page_content == "Section A"


def test_profile_uses_hierarchical_index_flag_and_retriever() -> None:
    assert profile_uses_hierarchical_index(RagProfile(hierarchical_index_enabled=True)) is True
    assert profile_uses_hierarchical_index(RagProfile(retriever_id="hierarchical")) is True
    assert profile_uses_hierarchical_index(RagProfile()) is False


@pytest.mark.no_ci
def test_rag_stack_wires_toc_store_when_hierarchical_enabled() -> None:
    from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack

    profile = RagProfile(
        hierarchical_index_enabled=True,
        retriever_id="hierarchical",
        enable_rerank=False,
        route_mode="off",
    )
    stack = create_default_rag_stack(profile=profile)
    assert stack.toc_vectorstore_manager is not None
    assert stack.toc_vectorstore_manager is not stack.vectorstore_manager


@pytest.mark.no_ci
def test_dual_index_native_ingest_with_parent_child_chunks(tmp_path) -> None:
    source = tmp_path / "book.txt"
    book_text = "alpha " * 200 + "beta " * 200 + "gamma " * 200
    source.write_text(book_text, encoding="utf-8")

    chunks_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    toc_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    embedding = build_fake_embedding_manager()
    strategy = ParentChildChunkingStrategy(parent_size=400, child_size=80, child_overlap=10)

    pipeline = IngestPipeline(
        loader=_InlineLoader(
            [
                KnowledgeDocument.model_validate(
                    {
                        "schema_version": 1,
                        "identity": {
                            "document_id": "book-1",
                            "root_document_id": "book-1",
                        },
                        "scope": {"tenant_id": "book"},
                        "content": book_text,
                        "metadata": {"doc_id": "book-1"},
                        "provenance": {
                            "source_kind": "file",
                            "source_id": str(source),
                        },
                    }
                )
            ]
        ),
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


@pytest.mark.no_ci
def test_dual_index_ingest_and_hierarchical_retrieve_expands_parents(tmp_path) -> None:
    from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
    from intergrax.rag.retrieval.retrieval_service import RetrievalService
    from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
    from intergrax.rag.retrievers.providers.hierarchical_retriever import HierarchicalRetriever

    source = tmp_path / "book.txt"
    book_text = "alpha " * 200 + "beta " * 200 + "gamma " * 200
    source.write_text(book_text, encoding="utf-8")

    chunks_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    toc_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="book"))
    embedding = build_fake_embedding_manager()
    strategy = ParentChildChunkingStrategy(parent_size=400, child_size=80, child_overlap=10)

    pipeline = IngestPipeline(
        loader=_InlineLoader(
            [
                KnowledgeDocument.model_validate(
                    {
                        "schema_version": 1,
                        "identity": {
                            "document_id": "book-1",
                            "root_document_id": "book-1",
                        },
                        "scope": {"tenant_id": "book"},
                        "content": book_text,
                        "metadata": {"doc_id": "book-1"},
                        "provenance": {
                            "source_kind": "file",
                            "source_id": str(source),
                        },
                    }
                )
            ]
        ),
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
