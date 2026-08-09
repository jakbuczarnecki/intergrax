from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
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
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreHit,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

SCOPE = VectorStoreScope(
    tenant_id="dual-reingest-tenant",
    namespace="hierarchical",
    workspace_id="workspace-a",
)


class _FileLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del kwargs
        source_path = Path(source).resolve()
        source_id = str(source_path)
        root_id = "root-" + hashlib.sha256(source_id.encode()).hexdigest()[:16]
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": root_id,
                        "root_document_id": root_id,
                    },
                    "scope": {
                        "tenant_id": SCOPE.tenant_id,
                        "namespace": SCOPE.namespace,
                    },
                    "content": source_path.read_text(encoding="utf-8"),
                    "metadata": {},
                    "provenance": {
                        "source_kind": "file",
                        "source_id": source_id,
                    },
                }
            )
        ]


class _ParentChildSplitter:
    def __init__(self) -> None:
        self._strategy = ParentChildChunkingStrategy(
            parent_size=100,
            child_size=50,
            child_overlap=0,
        )

    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> Sequence[KnowledgeDocument]:
        del strategy_id
        return self._strategy.chunk(documents)


class _DeterministicEmbeddingManager(BaseEmbeddingManager):
    dimension = 2

    def embed_one(self, text: str) -> np.ndarray:
        return self._vector(text)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        native_documents = tuple(documents)
        return EmbeddingResult(
            documents=native_documents,
            embeddings=self.embed_texts(
                [document.content for document in native_documents]
            ),
        )

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        if "A-v3-target" in text or text.startswith("A-v3-target"):
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if "parent_" in text:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 1.0], dtype=np.float32)


def _group(label: str) -> str:
    return label.ljust(100, "_")


def _write(path: Path, *groups: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_group(group) for group in groups), encoding="utf-8")


def _build() -> tuple[
    IngestPipeline,
    VectorstoreManager,
    VectorstoreManager,
    RetrievalService,
]:
    profile = RagProfile(
        retriever_id="hierarchical",
        fast_retriever_id="hierarchical",
        deep_retriever_id="hierarchical",
        hierarchical_index_enabled=True,
        chunking_strategy_id="parent_child",
        enable_rerank=False,
        route_mode="off",
        native_hybrid_enabled=False,
        query_expansion="off",
    )
    embedding = _DeterministicEmbeddingManager()
    main = VectorstoreManager(
        InMemoryVectorStore(tenant_id=SCOPE.tenant_id),
        scope=SCOPE,
    )
    toc = VectorstoreManager(
        InMemoryVectorStore(tenant_id=SCOPE.tenant_id),
        scope=SCOPE,
    )
    pipeline = IngestPipeline(
        loader=_FileLoader(),
        splitter=_ParentChildSplitter(),
        embedding_manager=embedding,
        vectorstore=main,
        toc_vectorstore=toc,
        profile=profile,
    )
    retrieval = RetrievalService(
        retriever_manager=create_default_retriever_manager(
            vector_store=main,
            embedding_manager=embedding,
            toc_vector_store=toc,
            profile=profile,
            discover_entry_points=False,
        ),
        profile=profile,
    )
    return pipeline, main, toc, retrieval


def _ingest(pipeline: IngestPipeline, path: Path) -> list[str]:
    result = pipeline.run(
        IngestRequest(
            source_path=str(path),
            base_metadata={
                "tenant_id": SCOPE.tenant_id,
                "namespace": SCOPE.namespace,
            },
            workspace_id=SCOPE.workspace_id,
            chunking_strategy_id="parent_child",
        )
    )
    assert result.used is True
    assert result.reason == "ok"
    return result.vector_ids


def _source_ids(store: VectorstoreManager, path: Path) -> set[str]:
    return set(
        store.list_source_record_ids(
            source_id=str(path.resolve()),
            scope=SCOPE,
        )
    )


def _source_hits(store: VectorstoreManager, path: Path) -> list[VectorStoreHit]:
    source_id = str(path.resolve())
    return [
        hit
        for hit in store.query(
            query_embedding=[1.0, 0.0],
            scope=SCOPE,
            top_k=100,
        )
        if str(hit.document.provenance.source_id) == source_id
    ]


def test_dual_index_reingest_removes_stale_main_and_toc_records(
    tmp_path: Path,
) -> None:
    pipeline, main, toc, retrieval = _build()
    source_a = tmp_path / "a" / "same.txt"
    source_b = tmp_path / "b" / "same.txt"

    _write(source_a, "A-v1-target", "A-v1-tail")
    _write(source_b, "B-stable")
    a_v1_main = _source_ids(main, source_a)
    a_v1_toc = _source_ids(toc, source_a)
    assert not a_v1_main
    assert not a_v1_toc
    _ingest(pipeline, source_a)
    _ingest(pipeline, source_b)
    a_v1_main = _source_ids(main, source_a)
    a_v1_toc = _source_ids(toc, source_a)
    b_main = _source_ids(main, source_b)
    b_toc = _source_ids(toc, source_b)
    assert len(a_v1_main) == 4
    assert len(a_v1_toc) == 2

    _write(source_a, "A-v2-target", "A-v2-tail")
    _ingest(pipeline, source_a)
    a_v2_main = _source_ids(main, source_a)
    a_v2_toc = _source_ids(toc, source_a)
    stale_v1_main = a_v1_main - a_v2_main
    stale_v1_toc = a_v1_toc - a_v2_toc
    assert len(stale_v1_main) == 2
    assert len(stale_v1_toc) == 2
    assert stale_v1_main.isdisjoint(a_v2_main)
    assert stale_v1_toc.isdisjoint(a_v2_toc)
    assert all("A-v1-" not in hit.content for hit in _source_hits(main, source_a))
    assert _source_ids(main, source_b) == b_main
    assert _source_ids(toc, source_b) == b_toc

    stale_tail_main = {
        hit.vector_id
        for hit in _source_hits(main, source_a)
        if "A-v2-tail" in hit.content
    }
    stale_tail_toc = {
        hit.vector_id
        for hit in _source_hits(toc, source_a)
        if hit.content.endswith("parent_1")
    }
    assert stale_tail_main
    assert stale_tail_toc

    _write(source_a, "A-v3-target")
    _ingest(pipeline, source_a)
    a_v3_main = _source_ids(main, source_a)
    a_v3_toc = _source_ids(toc, source_a)
    assert stale_tail_main.isdisjoint(a_v3_main)
    assert stale_tail_toc.isdisjoint(a_v3_toc)
    assert len(a_v3_main) == 2
    assert len(a_v3_toc) == 1
    assert all(
        marker not in hit.content
        for hit in _source_hits(main, source_a)
        for marker in ("A-v1-", "A-v2-", "A-v2-tail")
    )
    assert main.count(scope=SCOPE) == len(a_v3_main) + len(b_main)
    assert toc.count(scope=SCOPE) == len(a_v3_toc) + len(b_toc)
    assert _source_ids(main, source_b) == b_main
    assert _source_ids(toc, source_b) == b_toc

    main_before_repeat = _source_ids(main, source_a)
    toc_before_repeat = _source_ids(toc, source_a)
    _ingest(pipeline, source_a)
    assert _source_ids(main, source_a) == main_before_repeat
    assert _source_ids(toc, source_a) == toc_before_repeat
    assert main.count(scope=SCOPE) == len(a_v3_main) + len(b_main)
    assert toc.count(scope=SCOPE) == len(a_v3_toc) + len(b_toc)

    current_a = retrieval.retrieve(
        RetrievalRequest(
            query="A-v3-target",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=10,
            route_tier_override="standard",
        )
    )
    assert current_a.used is True
    assert any("A-v3-target" in chunk.text for chunk in current_a.chunks)
    assert all("A-v1-" not in chunk.text for chunk in current_a.chunks)
    assert all("A-v2-" not in chunk.text for chunk in current_a.chunks)
    assert all(
        (
            chunk.scope["tenant_id"],
            chunk.scope["namespace"],
            chunk.scope["workspace_id"],
        )
        == (SCOPE.tenant_id, SCOPE.namespace, SCOPE.workspace_id)
        for chunk in current_a.chunks
    )

    current_b = retrieval.retrieve(
        RetrievalRequest(
            query="B-stable",
            scope=SCOPE,
            final_top_k=5,
            prefetch_k=10,
            route_tier_override="standard",
        )
    )
    assert current_b.used is True
    assert any("B-stable" in chunk.text for chunk in current_b.chunks)
