from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.graph.contracts.graph_store import GraphNode
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

TENANT_ID = "graph-reingest-10b-tenant"
NAMESPACE = "graph-reingest-10b"
WORKSPACE_ID = "graph-reingest-10b-workspace"
SCOPE = VectorStoreScope(
    tenant_id=TENANT_ID,
    namespace=NAMESPACE,
    workspace_id=WORKSPACE_ID,
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
                        "tenant_id": TENANT_ID,
                        "namespace": NAMESPACE,
                        "workspace_id": WORKSPACE_ID,
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


class _FileSplitter:
    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> list[KnowledgeDocument]:
        del strategy_id
        chunks: list[KnowledgeDocument] = []
        for document in documents:
            chunks.extend(
                build_derived_chunk(
                    document,
                    content=content.strip(),
                    strategy_id="graph-reingest-10b",
                    chunk_index=index,
                )
                for index, content in enumerate(document.content.split("|"))
                if content.strip()
            )
        return chunks


class _MarkerEmbeddingManager(BaseEmbeddingManager):
    dimension = 4

    def __init__(self) -> None:
        self.fail_on: str | None = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.fail_on and any(self.fail_on in text for text in texts):
            raise RuntimeError("controlled embedding failure")
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            if "A-old" in text:
                vectors[index, 0] = 1.0
            elif "A-new" in text:
                vectors[index, 2] = 1.0
            elif "B-stable" in text:
                vectors[index, 1] = 1.0
            elif "Shared" in text:
                vectors[index, 3] = 1.0
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

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


class _FaultGraphStore(InMemoryGraphStore):
    def __init__(self) -> None:
        super().__init__(tenant_id=TENANT_ID)
        self.fail_index = False
        self.fail_unlink = False

    def upsert_node(self, node: GraphNode) -> None:
        if self.fail_index:
            raise RuntimeError("controlled graph indexing failure")
        super().upsert_node(node)

    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        if self.fail_unlink:
            raise RuntimeError("controlled graph unlink failure")
        return super().unlink_chunks(chunk_ids)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build() -> tuple[
    IngestPipeline,
    _MarkerEmbeddingManager,
    VectorstoreManager,
    _FaultGraphStore,
    RetrievalService,
]:
    profile = RagProfile(
        retriever_id="graph_rag",
        fast_retriever_id="graph_rag",
        deep_retriever_id="graph_rag",
        graph_rag_enabled=True,
        graph_store_backend="inmemory",
        enable_rerank=False,
        route_mode="off",
        native_hybrid_enabled=False,
    )
    embedding = _MarkerEmbeddingManager()
    vectorstore = VectorstoreManager(
        InMemoryVectorStore(tenant_id=TENANT_ID),
        scope=SCOPE,
    )
    graph = _FaultGraphStore()
    pipeline = IngestPipeline(
        loader=_FileLoader(),
        splitter=_FileSplitter(),
        embedding_manager=embedding,
        vectorstore=vectorstore,
        profile=profile,
        graph_store=graph,
    )
    retrieval = RetrievalService(
        retriever_manager=create_default_retriever_manager(
            vector_store=vectorstore,
            embedding_manager=embedding,
            graph_store=graph,
            profile=profile,
        ),
        profile=profile,
    )
    return pipeline, embedding, vectorstore, graph, retrieval


def _ingest(pipeline: IngestPipeline, path: Path) -> list[str]:
    result = pipeline.run(
        IngestRequest(
            source_path=str(path),
            base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
            workspace_id=WORKSPACE_ID,
        )
    )
    assert result.used is True
    assert result.reason == "ok"
    return result.vector_ids


def _source_ids(vectorstore: VectorstoreManager, path: Path) -> set[str]:
    return set(
        vectorstore.list_source_record_ids(
            source_id=str(path.resolve()),
            scope=SCOPE,
        )
    )


def _retrieve(
    retrieval: RetrievalService,
    query: str,
):
    return retrieval.retrieve(
        RetrievalRequest(
            query=query,
            scope=SCOPE,
            final_top_k=10,
            prefetch_k=10,
            route_tier_override="standard",
        )
    )


def test_graphrag_source_reingest_replaces_vectors_and_graph_safely(
    tmp_path: Path,
) -> None:
    pipeline, embedding, vectorstore, graph, retrieval = _build()
    source_a = tmp_path / "a" / "same.txt"
    source_b = tmp_path / "b" / "same.txt"

    _write(
        source_a,
        "A-old Old Alpha supports Old Beta|"
        "A-old Old Alpha supports Shared Entity|"
        "A-old Shared One connects Shared Two",
    )
    _write(
        source_b,
        "B-stable Shared Entity supports Shared One|"
        "B-stable Shared Two supports B Stable",
    )
    a_v1 = set(_ingest(pipeline, source_a))
    b_v1 = set(_ingest(pipeline, source_b))
    b_graph_nodes = graph.node_ids_for_chunks(b_v1)
    assert len(a_v1) == 3
    assert len(b_v1) == 2
    a_v1_graph_nodes = graph.node_ids_for_chunks(a_v1)
    assert a_v1_graph_nodes
    assert a_v1 <= set(graph.chunk_ids_for_nodes(a_v1_graph_nodes))

    _write(source_a, "A-failing Old Alpha supports Old Beta")
    embedding.fail_on = "A-failing"
    with pytest.raises(RuntimeError, match="controlled embedding failure"):
        pipeline.run(
            IngestRequest(
                source_path=str(source_a),
                base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
                workspace_id=WORKSPACE_ID,
            )
        )
    embedding.fail_on = None
    assert _source_ids(vectorstore, source_a) == a_v1
    assert graph.node_ids_for_chunks(a_v1)

    _write(
        source_a,
        "A-new New Alpha supports Shared Entity|A-new New Tail supports New Alpha",
    )
    graph.fail_index = True
    with pytest.raises(RuntimeError, match="source_reingest_graph_publish_failed"):
        pipeline.run(
            IngestRequest(
                source_path=str(source_a),
                base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
                workspace_id=WORKSPACE_ID,
            )
        )
    graph.fail_index = False
    a_v2_partial = _source_ids(vectorstore, source_a)
    assert a_v1.issubset(a_v2_partial)
    assert len(a_v2_partial) > len(a_v1)
    assert graph.node_ids_for_chunks(a_v1)

    a_v2 = set(_ingest(pipeline, source_a))
    assert a_v1.isdisjoint(a_v2)
    assert _source_ids(vectorstore, source_a) == a_v2
    assert graph.node_ids_for_chunks(a_v1) == set()
    assert graph.find_nodes(label_contains="Old", limit=10) == []
    assert graph.node_ids_for_chunks(b_v1) == b_graph_nodes
    assert "ent:shared_two" not in {
        node.id for node in graph.neighbors("ent:shared_one")
    }

    _write(source_a, "A-new New Alpha remains")
    graph.fail_unlink = True
    with pytest.raises(RuntimeError, match="source_reingest_graph_stale_unlink_failed"):
        pipeline.run(
            IngestRequest(
                source_path=str(source_a),
                base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
                workspace_id=WORKSPACE_ID,
            )
        )
    graph.fail_unlink = False
    a_v3 = set(_ingest(pipeline, source_a))
    assert len(a_v3) == 1
    assert a_v2 - a_v3
    assert graph.node_ids_for_chunks(a_v2 - a_v3) == set()
    assert graph.node_ids_for_chunks(b_v1) == b_graph_nodes
    assert graph.find_nodes(label_contains="Old", limit=10) == []

    repeated = set(_ingest(pipeline, source_a))
    assert repeated == a_v3
    assert _source_ids(vectorstore, source_a) == a_v3
    assert graph.node_ids_for_chunks(b_v1) == b_graph_nodes

    current_a = _retrieve(retrieval, "A-new New Alpha")
    assert current_a.used is True
    assert any("New Alpha" in chunk.text for chunk in current_a.chunks)
    assert all("Old Alpha" not in chunk.text for chunk in current_a.chunks)
    assert all(
        chunk.scope.get("tenant_id") == TENANT_ID
        and chunk.scope.get("namespace") == NAMESPACE
        and chunk.scope.get("workspace_id") == WORKSPACE_ID
        for chunk in current_a.chunks
    )

    old_a = _retrieve(retrieval, "A-old Old Alpha")
    assert all("Old Alpha" not in chunk.text for chunk in old_a.chunks)
    assert all("Old Beta" not in chunk.text for chunk in old_a.chunks)

    current_b = _retrieve(retrieval, "B-stable Shared Entity")
    assert current_b.used is True
    assert any("B-stable" in chunk.text for chunk in current_b.chunks)
    assert any("Shared Entity" in chunk.text for chunk in current_b.chunks)
    assert all(
        chunk.scope.get("tenant_id") == TENANT_ID
        and chunk.scope.get("namespace") == NAMESPACE
        and chunk.scope.get("workspace_id") == WORKSPACE_ID
        for chunk in current_b.chunks
    )
