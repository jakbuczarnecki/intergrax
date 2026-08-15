from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.distributed.source_operation import (
    DocumentStoreSourceOperationCoordinator,
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SourceOperationLease,
    SourceOperationCoordinator,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

TENANT_ID = "same-source-10c-tenant"
NAMESPACE = "same-source-10c"
WORKSPACE_ID = "same-source-10c-workspace"
SOURCE_ID = "canonical-source-id"
SCOPE = VectorStoreScope(
    tenant_id=TENANT_ID,
    namespace=NAMESPACE,
    workspace_id=WORKSPACE_ID,
)


class _ThreadVersionLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del kwargs
        version = threading.current_thread().name
        root_id = "root-" + hashlib.sha256(SOURCE_ID.encode()).hexdigest()[:16]
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {"document_id": root_id, "root_document_id": root_id},
                    "scope": {
                        "tenant_id": TENANT_ID,
                        "namespace": NAMESPACE,
                        "workspace_id": WORKSPACE_ID,
                    },
                    "content": f"{version}-content",
                    "metadata": {},
                    "provenance": {"source_kind": "test", "source_id": SOURCE_ID},
                }
            )
        ]


class _Splitter:
    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> list[KnowledgeDocument]:
        del strategy_id
        return [
            build_derived_chunk(
                document,
                content=document.content,
                strategy_id="same-source-10c",
                chunk_index=0,
            )
            for document in documents
        ]


class _Embedding(BaseEmbeddingManager):
    dimension = 2

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0] if "T1" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        native_documents = tuple(documents)
        return EmbeddingResult(
            documents=native_documents,
            embeddings=self.embed_texts([document.content for document in native_documents]),
        )


GRAPH_SOURCE_A = "graph-race-source-a"
GRAPH_SOURCE_B = "graph-race-source-b"


class _GraphRaceLoader:
    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del kwargs
        source_path = Path(source).resolve()
        source_id = GRAPH_SOURCE_A if source_path.name == "source-a.txt" else GRAPH_SOURCE_B
        if source_id == GRAPH_SOURCE_A:
            content = (
                "Old Entity supports Shared Entity"
                if threading.current_thread().name == "G1"
                else "New Entity supports Shared Entity"
            )
        else:
            content = "Source B supports Shared Entity"
        root_id = "root-" + hashlib.sha256(source_id.encode()).hexdigest()[:16]
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {"document_id": root_id, "root_document_id": root_id},
                    "scope": {
                        "tenant_id": TENANT_ID,
                        "namespace": NAMESPACE,
                        "workspace_id": WORKSPACE_ID,
                    },
                    "content": content,
                    "metadata": {},
                    "provenance": {"source_kind": "test", "source_id": source_id},
                }
            )
        ]


class _GraphRaceSplitter:
    def split_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        strategy_id: str | None = None,
    ) -> list[KnowledgeDocument]:
        del strategy_id
        return [
            build_derived_chunk(
                document,
                content=document.content,
                strategy_id="graph-race-10c",
                chunk_index=0,
            )
            for document in documents
        ]


class _GraphRaceEmbedding(_Embedding):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0]
                if "Old Entity" in text
                else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class _BlockingGraphStore(InMemoryGraphStore):
    def __init__(self) -> None:
        super().__init__(tenant_id=TENANT_ID)
        self.write_started = threading.Event()
        self.allow_g1_write = threading.Event()

    def upsert_node(self, node) -> None:
        if threading.current_thread().name == "G1" and not self.write_started.is_set():
            self.write_started.set()
            assert self.allow_g1_write.wait(timeout=5)
        super().upsert_node(node)


class _SnapshotRaceVectorstore(VectorstoreManager):
    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore(tenant_id=TENANT_ID), scope=SCOPE)


class _BlockingPublicationVectorstore(_SnapshotRaceVectorstore):
    def __init__(self) -> None:
        super().__init__()
        self.write_started = threading.Event()
        self.allow_t1_write = threading.Event()

    def add_records(self, records, *, scope=None):
        if threading.current_thread().name == "T1":
            self.write_started.set()
            assert self.allow_t1_write.wait(timeout=5)
        return super().add_records(records, scope=scope)


class _ControlledCoordinator:
    def __init__(self) -> None:
        self._delegate = InProcessSourceOperationCoordinator()
        self.first_acquired = threading.Event()
        self.conflict_observed = threading.Event()
        self.allow_first = threading.Event()

    def acquire(self, *, key: RagSourceOperationKey) -> SourceOperationLease | None:
        lease = self._delegate.acquire(key=key)
        if lease is not None and not self.first_acquired.is_set():
            self.first_acquired.set()
            self.allow_first.wait(timeout=5)
        elif lease is None:
            self.conflict_observed.set()
        return lease

    def release(self, *, lease: SourceOperationLease) -> None:
        self._delegate.release(lease=lease)

    def is_owned(self, *, lease: SourceOperationLease) -> bool:
        return self._delegate.is_owned(lease=lease)

    def publication_generation(self, *, lease: SourceOperationLease) -> str:
        return self._delegate.publication_generation(lease=lease)

    def active_publication_generation(self, *, key: RagSourceOperationKey) -> str | None:
        return self._delegate.active_publication_generation(key=key)

    def promote_publication(self, *, lease: SourceOperationLease) -> bool:
        return self._delegate.promote_publication(lease=lease)


def _pipeline(
    vectorstore: VectorstoreManager,
    coordinator: SourceOperationCoordinator | None = None,
) -> IngestPipeline:
    return IngestPipeline(
        loader=_ThreadVersionLoader(),
        splitter=_Splitter(),
        embedding_manager=_Embedding(),
        vectorstore=vectorstore,
        profile=RagProfile(
            retriever_id="vector_similarity",
            fast_retriever_id="vector_similarity",
            deep_retriever_id="vector_similarity",
            enable_rerank=False,
            route_mode="off",
            native_hybrid_enabled=False,
        ),
        source_coordinator=coordinator,
    )


def _graph_pipeline(
    vectorstore: VectorstoreManager,
    graph: InMemoryGraphStore,
    coordinator: SourceOperationCoordinator,
) -> IngestPipeline:
    return IngestPipeline(
        loader=_GraphRaceLoader(),
        splitter=_GraphRaceSplitter(),
        embedding_manager=_GraphRaceEmbedding(),
        vectorstore=vectorstore,
        graph_store=graph,
        profile=RagProfile(
            retriever_id="graph_rag",
            fast_retriever_id="graph_rag",
            deep_retriever_id="graph_rag",
            graph_rag_enabled=True,
            graph_store_backend="inmemory",
            enable_rerank=False,
            route_mode="off",
            native_hybrid_enabled=False,
        ),
        source_coordinator=coordinator,
    )


def test_same_source_concurrent_reingest_serializes_to_one_version(
    tmp_path: Path,
) -> None:
    source = tmp_path / "same-source.txt"
    source.write_text("unused", encoding="utf-8")
    vectorstore = _SnapshotRaceVectorstore()
    initial = _pipeline(vectorstore).run(
        IngestRequest(
            source_path=str(source),
            base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
            workspace_id=WORKSPACE_ID,
        )
    )
    assert initial.reason == "ok"
    coordinator = _ControlledCoordinator()
    pipeline = _pipeline(vectorstore, coordinator)
    results: list[object] = []

    def _run() -> None:
        try:
            results.append(
                pipeline.run(
                    IngestRequest(
                        source_path=str(source),
                        base_metadata={
                            "tenant_id": TENANT_ID,
                            "namespace": NAMESPACE,
                        },
                        workspace_id=WORKSPACE_ID,
                    )
                )
            )
        except BaseException as exc:
            results.append(exc)

    threads = [
        threading.Thread(target=_run, name="T1"),
        threading.Thread(target=_run, name="T2"),
    ]
    for thread in threads:
        thread.start()
    assert coordinator.first_acquired.wait(timeout=5)
    assert coordinator.conflict_observed.wait(timeout=5)
    coordinator.allow_first.set()
    for thread in threads:
        thread.join(timeout=10)

    assert len(results) == 2
    reasons = [
        result.reason if not isinstance(result, BaseException) else None
        for result in results
    ]
    assert reasons.count("ok") == 1
    assert reasons.count("source_ingest_conflict") == 1
    records = [
        hit
        for hit in vectorstore.query(
            [0.70710677, 0.70710677],
            scope=SCOPE,
            top_k=10,
        )
        if str(hit.document.provenance.source_id) == SOURCE_ID
    ]
    assert len(records) == 1
    assert {record.document.content for record in records} in (
        {"T1-content"},
        {"T2-content"},
    )


def test_stale_inflight_publication_cannot_become_active_after_takeover(
    tmp_path: Path,
) -> None:
    source = tmp_path / "same-source.txt"
    source.write_text("unused", encoding="utf-8")
    vectorstore = _BlockingPublicationVectorstore()
    document_store = InMemoryDocumentStore()
    clock = {"now": 100.0}
    key = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE_ID,
        source_id=SOURCE_ID,
    )
    first_coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-1",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: "generation-1-token",
        version_factory=lambda: "lease-version-1",
    )
    second_coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-2",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: "generation-2-token",
        version_factory=lambda: "lease-version-2",
    )
    first_pipeline = _pipeline(vectorstore, first_coordinator)
    second_pipeline = _pipeline(vectorstore, second_coordinator)
    results: dict[str, object] = {}

    def _run(name: str, pipeline: IngestPipeline) -> None:
        try:
            results[name] = pipeline.run(
                IngestRequest(
                    source_path=str(source),
                    base_metadata={
                        "tenant_id": TENANT_ID,
                        "namespace": NAMESPACE,
                    },
                    workspace_id=WORKSPACE_ID,
                )
            )
        except BaseException as exc:
            results[name] = exc

    first_thread = threading.Thread(
        target=_run,
        args=("T1", first_pipeline),
        name="T1",
    )
    first_thread.start()
    assert vectorstore.write_started.wait(timeout=5)

    clock["now"] = 106.0
    second_thread = threading.Thread(
        target=_run,
        args=("T2", second_pipeline),
        name="T2",
    )
    second_thread.start()
    second_thread.join(timeout=10)
    assert not second_thread.is_alive()
    assert results["T2"].reason == "ok"

    vectorstore.allow_t1_write.set()
    first_thread.join(timeout=10)
    assert not first_thread.is_alive()
    assert isinstance(results["T1"], RuntimeError)
    assert str(results["T1"]) == "source_ingest_conflict"

    hits = vectorstore.query(
        [0.70710677, 0.70710677],
        scope=SCOPE,
        top_k=10,
    )
    source_hits = [
        hit for hit in hits if str(hit.document.provenance.source_id) == SOURCE_ID
    ]
    assert [hit.document.content for hit in source_hits] == ["T2-content"]
    assert (
        second_coordinator.active_publication_generation(key=key)
        == source_hits[0].document.metadata[
            "__intergrax_source_publication_generation"
        ]
    )


def test_stale_graph_topology_is_inactive_after_generation_takeover(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source-a.txt"
    source_b = tmp_path / "source-b.txt"
    source_a.write_text("unused", encoding="utf-8")
    source_b.write_text("unused", encoding="utf-8")
    vectorstore = _SnapshotRaceVectorstore()
    graph = _BlockingGraphStore()
    document_store = InMemoryDocumentStore()
    clock = {"now": 100.0}
    first_coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="graph-worker-1",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: "graph-generation-1-token",
        version_factory=lambda: "graph-lease-version-1",
    )
    second_coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="graph-worker-2",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: "graph-generation-2-token",
        version_factory=lambda: "graph-lease-version-2",
    )
    first_pipeline = _graph_pipeline(vectorstore, graph, first_coordinator)
    second_pipeline = _graph_pipeline(vectorstore, graph, second_coordinator)
    b_result = second_pipeline.run(
        IngestRequest(
            source_path=str(source_b),
            base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
            workspace_id=WORKSPACE_ID,
        )
    )
    assert b_result.reason == "ok"

    results: dict[str, object] = {}

    def _run(name: str, pipeline: IngestPipeline) -> None:
        try:
            results[name] = pipeline.run(
                IngestRequest(
                    source_path=str(source_a),
                    base_metadata={"tenant_id": TENANT_ID, "namespace": NAMESPACE},
                    workspace_id=WORKSPACE_ID,
                )
            )
        except BaseException as exc:
            results[name] = exc

    first_thread = threading.Thread(
        target=_run,
        args=("G1", first_pipeline),
        name="G1",
    )
    first_thread.start()
    assert graph.write_started.wait(timeout=5)
    g1_ids = set(
        vectorstore.list_source_record_ids(
            source_id=GRAPH_SOURCE_A,
            scope=SCOPE,
        )
    )
    assert g1_ids

    clock["now"] = 106.0
    second_thread = threading.Thread(
        target=_run,
        args=("G2", second_pipeline),
        name="G2",
    )
    second_thread.start()
    second_thread.join(timeout=10)
    assert not second_thread.is_alive()
    assert results["G2"].reason == "ok"

    graph.allow_g1_write.set()
    first_thread.join(timeout=10)
    assert not first_thread.is_alive()
    assert isinstance(results["G1"], RuntimeError)
    assert str(results["G1"]) == "source_ingest_conflict"

    assert "ent:old_entity" in graph._nodes
    assert graph.node_ids_for_chunks(g1_ids) == set()
    assert graph.find_nodes(label_contains="Old", limit=10) == []
    assert all(
        node.id != "ent:old_entity"
        for node in graph.neighbors("ent:shared_entity", max_hops=1)
    )

    retriever = GraphRagRetriever(
        vectorstore,
        _GraphRaceEmbedding(),
        graph,
        seed_top_k=1,
        graph_hops=1,
        source_coordinator=second_coordinator,
    )
    old_hits = retriever.retrieve(
        RetrieverQuery(
            query_text="Old Entity",
            query_embedding=[1.0, 0.0],
            top_k=10,
            scope=SCOPE,
        )
    )
    assert retriever.last_graph_trace is not None
    assert "ent:old_entity" not in retriever.last_graph_trace.expanded_node_ids
    assert all("Old Entity" not in hit.document.content for hit in old_hits)

    new_hits = retriever.retrieve(
        RetrieverQuery(
            query_text="New Entity",
            query_embedding=[0.0, 1.0],
            top_k=10,
            scope=SCOPE,
        )
    )
    assert any("New Entity" in hit.document.content for hit in new_hits)

    shared_hits = retriever.retrieve(
        RetrieverQuery(
            query_text="Shared Entity",
            query_embedding=[0.0, 1.0],
            top_k=10,
            scope=SCOPE,
        )
    )
    assert any("Source B" in hit.document.content for hit in shared_hits)


def test_source_operation_keys_allow_independent_sources_and_scopes() -> None:
    coordinator = InProcessSourceOperationCoordinator()
    source_a = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE_ID,
        source_id="source-a",
    )
    source_b = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE_ID,
        source_id="source-b",
    )
    other_workspace = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id="workspace-b",
        source_id="source-a",
    )
    other_namespace = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace="namespace-b",
        workspace_id=WORKSPACE_ID,
        source_id="source-a",
    )

    leases = [
        coordinator.acquire(key=key)
        for key in (source_a, source_b, other_workspace, other_namespace)
    ]
    assert all(lease is not None for lease in leases)
    assert len({lease.key.storage_id for lease in leases if lease is not None}) == 4
    for lease in leases:
        assert lease is not None
        coordinator.release(lease=lease)


def test_durable_source_operation_fencing_blocks_stale_owner() -> None:
    clock = {"now": 100.0}
    tokens = iter(("token-1", "token-2"))
    versions = iter(("version-1", "version-2"))
    store = InMemoryDocumentStore()
    key = RagSourceOperationKey(
        tenant_id=TENANT_ID,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE_ID,
        source_id=SOURCE_ID,
    )
    first = DocumentStoreSourceOperationCoordinator(
        store,
        owner_id="worker-1",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
        version_factory=lambda: next(versions),
    )
    second = DocumentStoreSourceOperationCoordinator(
        store,
        owner_id="worker-2",
        ttl_seconds=5,
        clock=lambda: clock["now"],
        token_factory=lambda: "token-2",
        version_factory=lambda: "version-2",
    )

    lease_1 = first.acquire(key=key)
    assert lease_1 is not None
    clock["now"] = 106.0
    lease_2 = second.acquire(key=key)
    assert lease_2 is not None
    assert first.is_owned(lease=lease_1) is False
    first.release(lease=lease_1)
    assert second.is_owned(lease=lease_2) is True
    second.release(lease=lease_2)
