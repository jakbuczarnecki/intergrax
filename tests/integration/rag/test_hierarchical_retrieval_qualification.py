from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import (
    ChunkMetadataKey,
)
from intergrax.rag.document_splitters.strategies.parent_child_chunking_strategy import (
    ParentChildChunkingStrategy,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.retrievers.providers.hierarchical_retriever import (
    HierarchicalRetriever,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_QUERY = "target evidence query"
_EXPECTED = "EXPECTED_SIBLING_EVIDENCE"


class _InlineLoader:
    def __init__(self, document: KnowledgeDocument) -> None:
        self._document = document

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del source, kwargs
        return [self._document]


class _InlineSplitter:
    def __init__(self, strategy: ParentChildChunkingStrategy) -> None:
        self._strategy = strategy

    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> Sequence[KnowledgeDocument]:
        del strategy_id
        return self._strategy.chunk(documents)


class _ControlledEmbeddingManager(BaseEmbeddingManager):
    def embed_one(self, text: str) -> NDArray[np.float32]:
        return self._vector(text)

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        native_documents = tuple(documents)
        return EmbeddingResult(
            documents=native_documents,
            embeddings=np.asarray(
                [self._vector(document.content) for document in native_documents],
                dtype=np.float32,
            ),
        )

    @staticmethod
    def _vector(text: str) -> NDArray[np.float32]:
        if text == _QUERY:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if text == "qualification-book:parent_0":
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if text.startswith("TARGET_ANCHOR"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if text.startswith(_EXPECTED):
            return np.asarray([0.1, 0.995], dtype=np.float32)
        if text.startswith("DISTRACTOR"):
            return np.asarray([0.9, 0.436], dtype=np.float32)
        if text.startswith("qualification-book:parent_"):
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


def _segment(label: str) -> str:
    return label.ljust(50, "x")


def _source_document(content: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "qualification-book",
                "root_document_id": "qualification-book",
            },
            "scope": {
                "tenant_id": "tenant-hier",
                "namespace": "knowledge",
            },
            "content": content,
            "metadata": {"source": "controlled-hierarchical-fixture"},
            "provenance": {
                "source_kind": "file",
                "source_id": "qualification-book.txt",
            },
        }
    )


def _scope_values(scope: VectorStoreScope) -> tuple[str, str | None, str | None]:
    return scope.tenant_id, scope.namespace, scope.workspace_id


def test_hierarchical_retrieval_production_qualification(tmp_path: Path) -> None:
    source = tmp_path / "qualification-book.txt"
    groups = [
        _segment("TARGET_ANCHOR") + _segment(_EXPECTED),
        *(
            _segment(f"DISTRACTOR_{index}") + _segment(f"DISTRACTOR_{index}_B")
            for index in range(1, 12)
        ),
    ]
    source.write_text("".join(groups), encoding="utf-8")
    document = _source_document(source.read_text(encoding="utf-8"))
    embedding = _ControlledEmbeddingManager()
    chunks_store = VectorstoreManager(
        store=InMemoryVectorStore(tenant_id="tenant-hier")
    )
    toc_store = VectorstoreManager(store=InMemoryVectorStore(tenant_id="tenant-hier"))
    profile = RagProfile(
        retriever_id="hierarchical",
        fast_retriever_id="hierarchical",
        deep_retriever_id="hierarchical",
        hierarchical_index_enabled=True,
        chunking_strategy_id="parent_child",
        enable_rerank=False,
        route_mode="off",
    )

    pipeline = IngestPipeline(
        loader=_InlineLoader(document),
        splitter=_InlineSplitter(
            ParentChildChunkingStrategy(parent_size=100, child_size=50, child_overlap=0)
        ),
        embedding_manager=embedding,
        vectorstore=chunks_store,
        toc_vectorstore=toc_store,
        profile=profile,
    )
    ingest = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": "tenant-hier", "namespace": "knowledge"},
            workspace_id="workspace-a",
            chunking_strategy_id="parent_child",
        )
    )

    assert ingest.used is True
    assert ingest.num_chunks == 24
    scope = VectorStoreScope(
        tenant_id="tenant-hier",
        namespace="knowledge",
        workspace_id="workspace-a",
    )
    assert chunks_store.count(scope=scope) == 24
    assert toc_store.count(scope=scope) == 12

    all_chunks = chunks_store.query(
        query_embedding=embedding.embed_one(_QUERY),
        scope=scope,
        top_k=100,
    )
    all_toc = toc_store.query(
        query_embedding=embedding.embed_one(_QUERY),
        scope=scope,
        top_k=100,
    )
    assert len(all_chunks) == 24
    assert len(all_toc) == 12

    child_parent_ids = {
        str(hit.document.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value])
        for hit in all_chunks
    }
    assert len(child_parent_ids) == 12
    for hit in all_chunks:
        assert hit.document.identity.root_document_id == "qualification-book"
        assert _scope_values(hit.document.scope) == _scope_values(scope)
        assert hit.document.metadata[ChunkMetadataKey.SECTION.value] == hit.document.metadata[
            ChunkMetadataKey.PARENT_CHUNK_ID.value
        ]

    for toc_hit in all_toc:
        toc = toc_hit.document
        parent_id = str(toc.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value])
        assert toc.identity.root_document_id == "qualification-book"
        assert _scope_values(toc.scope) == _scope_values(scope)
        assert parent_id in child_parent_ids
        assert any(
            child.document.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value] == parent_id
            and child.document.metadata[ChunkMetadataKey.SECTION.value]
            == toc.metadata[ChunkMetadataKey.SECTION.value]
            for child in all_chunks
        )

    direct_hits = chunks_store.query(
        query_embedding=embedding.embed_one(_QUERY),
        scope=scope,
        top_k=3,
    )
    assert _EXPECTED not in {hit.document.content[: len(_EXPECTED)] for hit in direct_hits}

    registry = RetrieverRegistry()
    registry.register(
        HierarchicalRetriever(
            chunks_store=chunks_store,
            embedding_manager=embedding,
            toc_store=toc_store,
            k_chunks=3,
            k_toc=1,
            max_toc_parents=1,
        )
    )
    retriever_manager = create_default_retriever_manager(
        vector_store=chunks_store,
        embedding_manager=embedding,
        toc_vector_store=toc_store,
        profile=profile,
        registry=registry,
        discover_entry_points=False,
    )
    service = RetrievalService(
        retriever_manager=retriever_manager,
        profile=profile,
    )
    result = service.retrieve(
        RetrievalRequest(
            query=_QUERY,
            top_k=2,
            scope=scope,
        )
    )

    assert result.used is True
    assert result.trace.retriever_id == "hierarchical"
    assert any(_EXPECTED in chunk.text for chunk in result.chunks)
    assert all(
        (
            chunk.scope["tenant_id"],
            chunk.scope["namespace"],
            chunk.scope["workspace_id"],
        )
        == _scope_values(scope)
        for chunk in result.chunks
    )
    assert all(
        chunk.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value]
        == "qualification-book:parent_0"
        for chunk in result.chunks
    )
    assert all(
        chunk.metadata[ChunkMetadataKey.SECTION.value]
        == "qualification-book:parent_0"
        for chunk in result.chunks
    )
    target_toc_score = next(
        hit.similarity_score
        for hit in all_toc
        if hit.document.metadata[ChunkMetadataKey.PARENT_CHUNK_ID.value]
        == "qualification-book:parent_0"
    )
    expected_chunk = next(chunk for chunk in result.chunks if _EXPECTED in chunk.text)
    assert expected_chunk.score == target_toc_score

    repeat = service.retrieve(
        RetrievalRequest(
            query=_QUERY,
            top_k=2,
            scope=scope,
        )
    )
    assert [
        (chunk.id, chunk.text, chunk.score, chunk.rank)
        for chunk in result.chunks
    ] == [
        (chunk.id, chunk.text, chunk.score, chunk.rank)
        for chunk in repeat.chunks
    ]
