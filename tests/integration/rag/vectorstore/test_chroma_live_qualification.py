from __future__ import annotations

import hashlib
import inspect
import os
import uuid
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.vector_store.chroma.bundle import (
    create_chroma_integration,
)
from intergrax.integrations.providers.vector_store.chroma.rag_store import (
    ChromaVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.soak.prod_slo import SoakConfig, run_vectorstore_soak
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

RUN_ENV = "INTERGRAX_RUN_CHROMA_LIVE"
DIMENSION = 4
P95_THRESHOLD_MS = 5_000.0


@dataclass
class _LiveContext:
    run_id: str
    stores: dict[str, ChromaVectorStore]
    clients: dict[str, Any]
    owned: list[tuple[ChromaVectorStore, VectorStoreScope, str]] = field(
        default_factory=list
    )

    def own(
        self,
        store: ChromaVectorStore,
        scope: VectorStoreScope,
        source_id: str,
    ) -> None:
        item = (store, scope, source_id)
        if item not in self.owned:
            self.owned.append(item)

    def cleanup(self) -> None:
        errors: list[str] = []
        for store, scope, source_id in self.owned:
            try:
                ids = tuple(
                    store.list_source_record_ids(source_id=source_id, scope=scope)
                )
                if ids:
                    store.delete(ids, scope=scope)
                remaining = store.list_source_record_ids(
                    source_id=source_id,
                    scope=scope,
                )
                if remaining:
                    errors.append(f"{source_id}: remaining={tuple(remaining)}")
            except Exception as exc:  # noqa: BLE001 — cleanup must be reported
                errors.append(f"{source_id}: {type(exc).__name__}")

        for tenant, store in self.stores.items():
            try:
                client = self.clients[tenant]
                client.delete_collection(name=store.collection_name)
                names = {collection.name for collection in client.list_collections()}
                if store.collection_name in names:
                    errors.append(f"{tenant}: collection remains")
            except Exception as exc:  # noqa: BLE001 — cleanup must be reported
                errors.append(f"{tenant}: {type(exc).__name__}")

        if errors:
            pytest.fail("Chroma qualification cleanup failed: " + "; ".join(errors))


def _open_live_store(tenant_id: str, run_id: str) -> tuple[ChromaVectorStore, Any]:
    try:
        import chromadb

        with patch.object(chromadb, "HttpClient", wraps=chromadb.HttpClient) as http:
            integration = create_chroma_integration(
                collection_name=f"rag_live_15b_r2_{run_id}",
                tenant_id=tenant_id,
                mode="http",
                http_host=os.environ.get("INTERGRAX_CHROMA_HOST", "localhost"),
                http_port=int(os.environ.get("INTERGRAX_CHROMA_PORT", "8000")),
            )
    except (
        ImportError,
        IntegrationDependencyError,
        ConnectionError,
        TimeoutError,
    ) as exc:
        pytest.skip(f"Chroma backend unavailable during open: {type(exc).__name__}")

    store = integration.vector_store.rag_store
    assert isinstance(store, ChromaVectorStore)
    assert integration.config.mode == "http"
    assert http.call_count == 1
    assert type(store._client).__name__ == "Client"
    assert store._client.heartbeat()
    return store, store._client


@pytest.fixture
def live_context() -> Generator[_LiveContext, None, None]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the Chroma live qualification")

    run_id = uuid.uuid4().hex
    stores: dict[str, ChromaVectorStore] = {}
    clients: dict[str, Any] = {}
    try:
        for tenant in ("tenant-a", "tenant-b"):
            stores[tenant], clients[tenant] = _open_live_store(
                f"r2-{run_id}-{tenant}",
                run_id,
            )
    except BaseException:
        context = _LiveContext(run_id, stores, clients)
        context.cleanup()
        raise

    context = _LiveContext(run_id, stores, clients)
    try:
        yield context
    finally:
        context.cleanup()


def _record(
    vector_id: str,
    *,
    source_id: str,
    scope: VectorStoreScope,
    embedding: Sequence[float],
    metadata: dict[str, object] | None = None,
    content: str | None = None,
) -> VectorStoreRecord:
    document_id = f"document-{hashlib.sha256(vector_id.encode()).hexdigest()}"
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": content or f"content-{vector_id}",
            "metadata": metadata or {},
            "provenance": {
                "source_kind": "rag-live-15b-r2",
                "source_id": source_id,
            },
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=list(embedding),
        vector_id=vector_id,
    )


class _QualificationLoader:
    def __init__(self, scope: VectorStoreScope) -> None:
        self._scope = scope

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
                        "tenant_id": self._scope.tenant_id,
                        "namespace": self._scope.namespace,
                        "workspace_id": self._scope.workspace_id,
                    },
                    "content": source_path.read_text(encoding="utf-8"),
                    "metadata": {"file_name": source_path.name},
                    "provenance": {
                        "source_kind": "file",
                        "source_id": source_id,
                    },
                }
            )
        ]


class _QualificationSplitter:
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
                    strategy_id="rag-live-15b-r2",
                    chunk_index=index,
                )
                for index, content in enumerate(document.content.split("|"))
                if content.strip()
            )
        return chunks


class _QualificationEmbeddingManager(BaseEmbeddingManager):
    dimension = DIMENSION

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            if "A-old" in text:
                vectors[index, 0] = 1.0
            elif "A-new" in text:
                vectors[index, 1] = 1.0
            elif "B-stable" in text:
                vectors[index, 2] = 1.0
            else:
                vectors[index, 3] = 1.0
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        documents_tuple = tuple(documents)
        return EmbeddingResult(
            documents=documents_tuple,
            embeddings=self.embed_texts(
                [document.content for document in documents_tuple]
            ),
        )


def _build_pipeline(
    store: ChromaVectorStore,
    scope: VectorStoreScope,
) -> tuple[IngestPipeline, _QualificationEmbeddingManager, VectorstoreManager]:
    profile = RagProfile(
        retriever_id="vector_similarity",
        fast_retriever_id="vector_similarity",
        deep_retriever_id="vector_similarity",
        enable_rerank=False,
        route_mode="off",
        native_hybrid_enabled=False,
    )
    embedding = _QualificationEmbeddingManager()
    vectorstore = VectorstoreManager(store, scope=scope)
    pipeline = IngestPipeline(
        loader=_QualificationLoader(scope),
        splitter=_QualificationSplitter(),
        embedding_manager=embedding,
        vectorstore=vectorstore,
        profile=profile,
    )
    return pipeline, embedding, vectorstore


def _ingest(
    pipeline: IngestPipeline,
    source: Path,
    scope: VectorStoreScope,
) -> list[str]:
    result = pipeline.run(
        IngestRequest(
            source_path=str(source),
            base_metadata={
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
            },
            workspace_id=scope.workspace_id,
        )
    )
    assert result.used is True
    assert result.reason == "ok"
    return result.vector_ids


def _source_ids(
    store: ChromaVectorStore,
    source_id: str,
    scope: VectorStoreScope,
) -> set[str]:
    return set(store.list_source_record_ids(source_id=source_id, scope=scope))


def test_chroma_live_qualification(
    live_context: _LiveContext,
    tmp_path: Path,
) -> None:
    run_id = live_context.run_id
    store_a = live_context.stores["tenant-a"]
    store_b = live_context.stores["tenant-b"]

    tenant_a = VectorStoreScope(
        tenant_id=store_a.cfg.tenant_id,
        namespace=f"r2-{run_id}-namespace-a",
        workspace_id=f"r2-{run_id}-workspace-a",
    )
    tenant_a_other_namespace = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=f"r2-{run_id}-namespace-b",
        workspace_id=tenant_a.workspace_id,
    )
    tenant_a_other_workspace = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=tenant_a.namespace,
        workspace_id=f"r2-{run_id}-workspace-b",
    )
    tenant_a_reverse_scope = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=tenant_a_other_namespace.namespace,
        workspace_id=tenant_a_other_workspace.workspace_id,
    )
    tenant_b = VectorStoreScope(
        tenant_id=store_b.cfg.tenant_id,
        namespace=tenant_a.namespace,
        workspace_id=tenant_a.workspace_id,
    )
    scope_matrix = (
        (store_a, tenant_a),
        (store_a, tenant_a_other_namespace),
        (store_a, tenant_a_other_workspace),
        (store_a, tenant_a_reverse_scope),
        (store_b, tenant_b),
    )

    matrix_ids: dict[VectorStoreScope, set[str]] = {}
    for index, (store, scope) in enumerate(scope_matrix):
        source_id = f"matrix://{run_id}/{index}"
        valid_id = f"logical://{run_id}/matrix/{index}/valid"
        adversarial_id = f"logical://{run_id}/matrix/{index}/adversarial"
        matrix_ids[scope] = {valid_id, adversarial_id}
        live_context.own(store, scope, source_id)
        live_context.own(store, scope, f"{source_id}/adversarial")
        store.add_records(
            [
                _record(
                    valid_id,
                    source_id=source_id,
                    scope=scope,
                    embedding=[0.6, 0.8, 0.0, 0.0],
                ),
                _record(
                    adversarial_id,
                    source_id=f"{source_id}/adversarial",
                    scope=scope,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                ),
            ],
            scope=scope,
        )

    query_source = inspect.getsource(ChromaVectorStore.query)
    ownership_source = inspect.getsource(ChromaVectorStore.list_source_record_ids)
    assert "where=self._normalize_chroma_where(effective_where)" in query_source
    assert "MetadataFilter.for_scope(scope, metadata_filter)" in query_source
    assert "self._collection.get(" in ownership_source
    assert "query(" not in ownership_source

    for store, scope in scope_matrix:
        with patch.object(
            store._collection, "query", wraps=store._collection.query
        ) as query:
            hits = store.query(
                [1.0, 0.0, 0.0, 0.0],
                scope=scope,
                top_k=20,
            )
        assert set(hit.vector_id for hit in hits) == matrix_ids[scope]
        assert all(
            hit.document.scope.tenant_id == scope.tenant_id
            and hit.document.scope.namespace == scope.namespace
            and hit.document.scope.workspace_id == scope.workspace_id
            for hit in hits
        )
        assert store.count(scope=scope) == 2
        where = query.call_args.kwargs["where"]
        assert scope.tenant_id in str(where)
        assert scope.namespace in str(where)
        assert scope.workspace_id in str(where)

    with pytest.raises(VectorStoreContractError):
        store_a.query(
            [1.0, 0.0, 0.0, 0.0],
            scope=tenant_b,
            top_k=5,
        )

    identity_source = f"identity://{run_id}"
    identity_ids = [
        f"logical://{run_id}/identity/1",
        f"logical://{run_id}/identity/2",
    ]
    live_context.own(store_a, tenant_a, identity_source)
    identity_records = [
        _record(
            vector_id,
            source_id=identity_source,
            scope=tenant_a,
            embedding=[0.7, 0.7, 0.0, 0.0],
        )
        for vector_id in identity_ids
    ]
    added_ids = tuple(store_a.add_records(identity_records, scope=tenant_a))
    queried_ids = {
        hit.vector_id
        for hit in store_a.query(
            [0.7, 0.7, 0.0, 0.0],
            scope=tenant_a,
            top_k=20,
        )
        if hit.document.provenance.source_id == identity_source
    }
    with patch.object(
        store_a._collection,
        "get",
        wraps=store_a._collection.get,
    ) as ownership_get:
        owned_ids = tuple(
            store_a.list_source_record_ids(
                source_id=identity_source,
                scope=tenant_a,
            )
        )
    delete_ids = tuple(sorted(owned_ids))
    assert set(added_ids) == queried_ids == set(owned_ids) == set(identity_ids)
    assert ownership_get.call_args.kwargs["where"]
    assert identity_source in str(ownership_get.call_args.kwargs["where"])
    store_a.delete(delete_ids, scope=tenant_a)
    assert (
        store_a.list_source_record_ids(
            source_id=identity_source,
            scope=tenant_a,
        )
        == ()
    )

    metadata_source = f"metadata://{run_id}"
    live_context.own(store_a, tenant_a, metadata_source)
    metadata_id = f"logical://{run_id}/metadata/1"
    with pytest.raises(ValueError, match="reserved key"):
        _record(
            metadata_id,
            source_id=metadata_source,
            scope=tenant_a,
            embedding=[0.9, 0.1, 0.0, 0.0],
            metadata={
                "group": 7,
                "tenant_id": "foreign-tenant",
                "namespace": "foreign-namespace",
                "workspace_id": "foreign-workspace",
            },
        )
    store_a.add_records(
        [
            _record(
                metadata_id,
                source_id=metadata_source,
                scope=tenant_a,
                embedding=[0.9, 0.1, 0.0, 0.0],
                metadata={"group": 7},
            )
        ],
        scope=tenant_a,
    )
    with patch.object(
        store_a._collection,
        "query",
        wraps=store_a._collection.query,
    ) as metadata_query:
        filtered_hits = store_a.query(
            [0.9, 0.1, 0.0, 0.0],
            scope=tenant_a,
            top_k=20,
            metadata_filter=MetadataFilter(conditions={"group": 7}),
        )
    assert [hit.vector_id for hit in filtered_hits] == [metadata_id]
    filtered_document = filtered_hits[0].document
    assert filtered_document.content == f"content-{metadata_id}"
    assert filtered_document.provenance.source_id == metadata_source
    assert (
        filtered_document.scope.tenant_id,
        filtered_document.scope.namespace,
        filtered_document.scope.workspace_id,
    ) == (
        tenant_a.tenant_id,
        tenant_a.namespace,
        tenant_a.workspace_id,
    )
    assert filtered_document.metadata == {"group": 7}
    metadata_where = str(metadata_query.call_args.kwargs["where"])
    assert all(
        value in metadata_where for value in (tenant_a.tenant_id, tenant_a.namespace)
    )
    assert "group" in metadata_where
    with pytest.raises(VectorStoreContractError):
        MetadataFilter(conditions={"tenant_id": "foreign"})

    delete_scopes = (
        (store_a, tenant_a),
        (store_a, tenant_a_other_namespace),
        (store_a, tenant_a_other_workspace),
        (store_a, tenant_a_reverse_scope),
        (store_b, tenant_b),
    )
    delete_ids_by_scope: dict[VectorStoreScope, str] = {}
    for index, (store, scope) in enumerate(delete_scopes):
        source_id = f"delete://{run_id}/{index}"
        vector_id = f"logical://{run_id}/delete/{index}"
        delete_ids_by_scope[scope] = vector_id
        live_context.own(store, scope, source_id)
        store.add_records(
            [
                _record(
                    vector_id,
                    source_id=source_id,
                    scope=scope,
                    embedding=[0.5, 0.5, 0.0, 0.0],
                )
            ],
            scope=scope,
        )
    store_a.delete([delete_ids_by_scope[tenant_a]], scope=tenant_a)
    assert (
        store_a.list_source_record_ids(
            source_id=f"delete://{run_id}/0",
            scope=tenant_a,
        )
        == ()
    )
    for index, (store, scope) in enumerate(delete_scopes[1:], start=1):
        assert store.list_source_record_ids(
            source_id=f"delete://{run_id}/{index}",
            scope=scope,
        ) == (delete_ids_by_scope[scope],)

    pipeline, embedding, vectorstore = _build_pipeline(store_a, tenant_a)
    source_a = tmp_path / "a" / "same.txt"
    source_b = tmp_path / "b" / "same.txt"
    source_a_id = str(source_a.resolve())
    source_b_id = str(source_b.resolve())
    live_context.own(store_a, tenant_a, source_a_id)
    live_context.own(store_a, tenant_a, source_b_id)
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_text("A-old alpha|A-old tail", encoding="utf-8")
    source_b.write_text("B-stable beta", encoding="utf-8")
    baseline = vectorstore.count(scope=tenant_a)

    a_v1 = set(_ingest(pipeline, source_a, tenant_a))
    b_v1 = set(_ingest(pipeline, source_b, tenant_a))
    assert len(a_v1) == 2
    assert len(b_v1) == 1
    assert _source_ids(vectorstore, source_a_id, tenant_a) == a_v1
    assert _source_ids(vectorstore, source_b_id, tenant_a) == b_v1
    assert vectorstore.count(scope=tenant_a) == baseline + 3

    source_a.write_text("A-new alpha|A-new tail", encoding="utf-8")
    a_v2 = set(_ingest(pipeline, source_a, tenant_a))
    assert len(a_v2) == 2
    assert a_v1.isdisjoint(a_v2)
    assert _source_ids(vectorstore, source_a_id, tenant_a) == a_v2
    assert _source_ids(vectorstore, source_b_id, tenant_a) == b_v1

    source_a.write_text("A-new alpha", encoding="utf-8")
    a_v3 = set(_ingest(pipeline, source_a, tenant_a))
    assert len(a_v3) == 1
    assert _source_ids(vectorstore, source_a_id, tenant_a) == a_v3
    assert _source_ids(vectorstore, source_b_id, tenant_a) == b_v1
    assert vectorstore.count(scope=tenant_a) == baseline + 2

    repeated_v3 = set(_ingest(pipeline, source_a, tenant_a))
    assert repeated_v3 == a_v3
    assert _source_ids(vectorstore, source_a_id, tenant_a) == a_v3
    assert vectorstore.count(scope=tenant_a) == baseline + 2
    replacement_hits = vectorstore.query(
        embedding.embed_one("A-new"),
        scope=tenant_a,
        top_k=20,
    )
    a_hits = [
        hit
        for hit in replacement_hits
        if hit.document.provenance.source_id == source_a_id
    ]
    assert len(a_hits) == 1
    assert "A-new" in a_hits[0].document.content
    assert all("A-old" not in hit.document.content for hit in a_hits)

    invalid_dimension = _record(
        f"logical://{run_id}/invalid-dimension",
        source_id=f"failure://{run_id}/dimension",
        scope=tenant_a,
        embedding=[1.0, 0.0, 0.0],
    )
    live_context.own(store_a, tenant_a, f"failure://{run_id}/dimension")
    with pytest.raises((IntegrationConfigurationError, ValueError), match="dimension"):
        store_a.add_records([invalid_dimension], scope=tenant_a)

    with pytest.raises(TypeError, match="VectorStoreRecord"):
        store_a.add_records([object()], scope=tenant_a)  # type: ignore[list-item]

    with pytest.raises(RuntimeError, match="query backend failure"):
        with patch.object(
            store_a._collection,
            "query",
            side_effect=RuntimeError("query backend failure"),
        ):
            store_a.query([1.0, 0.0, 0.0, 0.0], scope=tenant_a, top_k=1)

    with pytest.raises(RuntimeError, match="ownership backend failure"):
        with patch.object(
            store_a._collection,
            "get",
            side_effect=RuntimeError("ownership backend failure"),
        ):
            store_a.list_source_record_ids(
                source_id=source_a_id,
                scope=tenant_a,
            )

    delete_failure_id = f"logical://{run_id}/delete-failure"
    delete_failure_source = f"failure://{run_id}/delete"
    live_context.own(store_a, tenant_a, delete_failure_source)
    store_a.add_records(
        [
            _record(
                delete_failure_id,
                source_id=delete_failure_source,
                scope=tenant_a,
                embedding=[0.2, 0.8, 0.0, 0.0],
            )
        ],
        scope=tenant_a,
    )
    with pytest.raises(RuntimeError, match="delete backend failure"):
        with patch.object(
            store_a._collection,
            "delete",
            side_effect=RuntimeError("delete backend failure"),
        ):
            store_a.delete([delete_failure_id], scope=tenant_a)

    soak_store, soak_client = _open_live_store(f"r2-{run_id}-soak", run_id)
    live_context.stores["soak"] = soak_store
    live_context.clients["soak"] = soak_client
    soak = run_vectorstore_soak(
        soak_store,
        slug="chroma",
        config=SoakConfig(
            document_count=50,
            query_rounds=5,
            embedding_dim=DIMENSION,
            max_p95_query_ms=P95_THRESHOLD_MS,
        ),
    )
    for index in range(50):
        live_context.own(
            soak_store,
            VectorStoreScope(tenant_id=soak_store.cfg.tenant_id),
            f"soak-doc-{index}",
        )
    assert soak.passed is True, soak.reason
    assert soak.documents_indexed == 50
    assert soak.queries_executed == 5
    assert soak.p95_query_ms <= P95_THRESHOLD_MS

    print(
        "CHROMA_R2 "
        f"run={run_id} scope_matrix=PASS tenant=PASS namespace=PASS "
        "workspace=PASS combined=PASS adversarial=PASS "
        "server_filter=PASS identity=PASS ownership=PASS replacement=PASS "
        "same_basename=PASS delete=PASS metadata=PASS reconstruction=PASS "
        "failure_semantics=PASS "
        f"soak_records={soak.documents_indexed} soak_rounds={soak.queries_executed} "
        f"soak_p95_ms={soak.p95_query_ms:.2f} threshold_ms={P95_THRESHOLD_MS:.2f}"
    )
