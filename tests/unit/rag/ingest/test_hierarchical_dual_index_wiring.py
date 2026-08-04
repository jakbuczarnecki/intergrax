# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.knowledge.contracts.document import RESERVED_METADATA_KEYS

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.bootstrap.hierarchical_bootstrap import profile_uses_hierarchical_index
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
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


def test_dual_index_strategy_uses_native_embedding_for_main_and_toc() -> None:
    class _EmbeddingSpy:
        def __init__(self) -> None:
            self.calls: list[tuple[KnowledgeDocument, ...]] = []

        def embed_documents(
            self,
            documents: Sequence[KnowledgeDocument],
        ) -> EmbeddingResult:
            native_documents = tuple(documents)
            self.calls.append(native_documents)
            return EmbeddingResult(
                documents=native_documents,
                embeddings=np.ones((len(native_documents), 2), dtype=np.float32),
            )

        def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
            raise AssertionError("dual indexing must not use embed_texts")

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_documents(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    documents = [
        _native_chunk("chunk-a", "root-a", "tenant-a", "Section A"),
        _native_chunk("chunk-b", "root-a", "tenant-a", "Section B"),
    ]
    embedding = _EmbeddingSpy()
    main_store = _Store()
    toc_store = _Store()

    DualIndexStrategy(toc_vectorstore=toc_store, batch_size=1).build_index(
        documents=documents,
        embed_manager=embedding,
        vectorstore=main_store,
    )

    assert [[document.content for document in call] for call in embedding.calls] == [
        ["chunk-a", "chunk-b"],
        ["Section A", "Section B"],
    ]
    assert all(
        isinstance(document, KnowledgeDocument)
        for call in embedding.calls
        for document in call
    )
    toc_documents = embedding.calls[1]
    assert toc_documents[0].identity.parent_document_id == "chunk-a"
    assert toc_documents[0].identity.root_document_id == "root-a"
    assert toc_documents[0].scope.tenant_id == "tenant-a"
    assert toc_documents[0].content == "Section A"
    assert toc_documents[0].provenance.content_hash == hashlib.sha256(
        b"Section A"
    ).hexdigest()
    assert toc_documents[0].metadata[ChunkMetadataKey.SECTION.value] == "Section A"
    assert toc_documents[0].metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value] == "chunk-a"
    assert RESERVED_METADATA_KEYS.isdisjoint(toc_documents[0].metadata)
    assert toc_store.calls[0]["documents"][0].page_content == "Section A"
    assert toc_store.calls[1]["documents"][0].page_content == "Section B"


def test_dual_index_strategy_rejects_invalid_main_embeddings_before_any_store() -> None:
    class _Embedding:
        def __init__(self) -> None:
            self.calls: list[tuple[KnowledgeDocument, ...]] = []

        def embed_documents(
            self,
            documents: Sequence[KnowledgeDocument],
        ) -> EmbeddingResult:
            self.calls.append(tuple(documents))
            raise ValueError("invalid main embedding result")

        def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
            raise AssertionError("dual indexing must not use embed_texts")

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_documents(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    documents = [
        _native_chunk("chunk-a", "root-a", "tenant-a", "Section A"),
        _native_chunk("chunk-b", "root-a", "tenant-a", "Section B"),
    ]
    embedding = _Embedding()
    main_store = _Store()
    toc_store = _Store()

    with pytest.raises(ValueError, match="invalid main"):
        DualIndexStrategy(toc_vectorstore=toc_store).build_index(
            documents=documents,
            embed_manager=embedding,
            vectorstore=main_store,
        )

    assert [[document.content for document in call] for call in embedding.calls] == [
        ["chunk-a", "chunk-b"]
    ]
    assert main_store.calls == []
    assert toc_store.calls == []


def test_dual_index_strategy_rejects_invalid_toc_embeddings_after_main_store() -> None:
    class _Embedding:
        def __init__(self) -> None:
            self.calls: list[tuple[KnowledgeDocument, ...]] = []

        def embed_documents(
            self,
            documents: Sequence[KnowledgeDocument],
        ) -> EmbeddingResult:
            native_documents = tuple(documents)
            self.calls.append(native_documents)
            if len(self.calls) == 2:
                raise ValueError("invalid TOC embedding result")
            return EmbeddingResult(
                documents=native_documents,
                embeddings=np.ones((len(native_documents), 2), dtype=np.float32),
            )

        def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
            raise AssertionError("dual indexing must not use embed_texts")

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_documents(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    documents = [
        _native_chunk("chunk-a", "root-a", "tenant-a", "Section A"),
        _native_chunk("chunk-b", "root-a", "tenant-a", "Section B"),
    ]
    embedding = _Embedding()
    main_store = _Store()
    toc_store = _Store()

    with pytest.raises(ValueError, match="invalid TOC"):
        DualIndexStrategy(toc_vectorstore=toc_store, batch_size=1).build_index(
            documents=documents,
            embed_manager=embedding,
            vectorstore=main_store,
        )

    assert [[document.content for document in call] for call in embedding.calls] == [
        ["chunk-a", "chunk-b"],
        ["Section A", "Section B"],
    ]
    assert [
        [document.page_content for document in call["documents"]]
        for call in main_store.calls
    ] == [["chunk-a"], ["chunk-b"]]
    assert toc_store.calls == []


def _native_chunk(
    document_id: str,
    root_document_id: str,
    tenant_id: str,
    section: str,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": root_document_id,
                "parent_document_id": root_document_id,
            },
            "scope": {"tenant_id": tenant_id, "namespace": "rag"},
            "content": document_id,
            "metadata": {
                ChunkMetadataKey.SECTION.value: section,
                "source_marker": document_id,
            },
            "provenance": {
                "source_kind": "test",
                "source_id": f"source-{root_document_id}",
            },
        }
    )


def test_toc_grouping_is_first_seen_and_scope_safe() -> None:
    class _Embedding:
        def __init__(self) -> None:
            self.calls: list[tuple[KnowledgeDocument, ...]] = []

        def embed_documents(
            self,
            documents: Sequence[KnowledgeDocument],
        ) -> EmbeddingResult:
            native_documents = tuple(documents)
            self.calls.append(native_documents)
            return EmbeddingResult(
                documents=native_documents,
                embeddings=np.ones((len(native_documents), 2), dtype=np.float32),
            )

    class _Store:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_documents(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    documents = [
        _native_chunk("chunk-a", "root-a", "tenant-a", "Shared"),
        _native_chunk("chunk-b", "root-a", "tenant-a", "Shared"),
        _native_chunk("chunk-c", "root-b", "tenant-b", "Shared"),
    ]
    embedding = _Embedding()
    main_store = _Store()
    toc_store = _Store()

    DualIndexStrategy(toc_vectorstore=toc_store).build_index(
        documents=documents,
        embed_manager=embedding,
        vectorstore=main_store,
    )

    toc_documents = embedding.calls[1]
    assert [document.identity.parent_document_id for document in toc_documents] == [
        "chunk-a",
        "chunk-c",
    ]
    assert len({document.identity.document_id for document in toc_documents}) == 2
    assert [document.identity.root_document_id for document in toc_documents] == [
        "root-a",
        "root-b",
    ]
    assert [document.scope.tenant_id for document in toc_documents] == [
        "tenant-a",
        "tenant-b",
    ]


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
