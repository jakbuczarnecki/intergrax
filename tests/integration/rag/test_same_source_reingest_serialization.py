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
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
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


class _SnapshotRaceVectorstore(VectorstoreManager):
    def __init__(self) -> None:
        super().__init__(InMemoryVectorStore(tenant_id=TENANT_ID), scope=SCOPE)


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


def _pipeline(
    vectorstore: VectorstoreManager,
    coordinator: _ControlledCoordinator | None = None,
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
    reasons = [getattr(result, "reason", None) for result in results]
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
