from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.vector_store.pgvector.bundle import (
    create_pgvector_vector_store,
)
from intergrax.integrations.providers.vector_store.pgvector.rag_store import (
    PgVectorRagStore,
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
from intergrax.rag.vectorstore.soak.prod_slo import (
    SoakConfig,
    run_vectorstore_soak,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

DIMENSION_ENV = "INTERGRAX_PGVECTOR_DIMENSION"
DSN_ENV = "INTERGRAX_PGVECTOR_DSN"
DEFAULT_DIMENSION = 4
P95_THRESHOLD_MS = 5_000.0


def _open_live_store(tenant_id: str) -> PgVectorRagStore:
    dsn = os.environ.get(DSN_ENV, "").strip()
    raw_dimension = os.environ.get(DIMENSION_ENV, "").strip()
    if not dsn or not raw_dimension:
        pytest.skip(
            f"PgVector qualification requires {DSN_ENV} and {DIMENSION_ENV}"
        )
    try:
        dimension = int(raw_dimension)
    except ValueError as exc:
        raise AssertionError(f"{DIMENSION_ENV} must be an integer") from exc

    try:
        integration = create_pgvector_vector_store(
            tenant_id=tenant_id,
            dimension=dimension,
        )
        store = integration.rag_store
    except (IntegrationDependencyError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"PgVector backend unavailable during open: {type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001 — classify only setup connectivity
        error_type = type(exc)
        if (
            error_type.__module__.startswith(("psycopg", "psycopg2"))
            and error_type.__name__ in {"OperationalError", "InterfaceError"}
        ):
            pytest.skip(f"PgVector backend unavailable during open: {error_type.__name__}")
        raise

    assert isinstance(store, PgVectorRagStore)
    assert store._dimension == dimension
    health = store.health()
    assert health.healthy is True, health.detail
    return store


def _record(
    vector_id: str,
    *,
    source_id: str,
    scope: VectorStoreScope,
    embedding: Sequence[float],
    metadata: dict[str, object] | None = None,
    content: str | None = None,
) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": f"document-{hashlib.sha256(vector_id.encode()).hexdigest()}",
                "root_document_id": f"document-{hashlib.sha256(vector_id.encode()).hexdigest()}",
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": content or f"content-{vector_id}",
            "metadata": metadata or {},
            "provenance": {
                "source_kind": "rag-live-15a-r2",
                "source_id": source_id,
            },
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=list(embedding),
        vector_id=vector_id,
    )


@dataclass
class _LiveContext:
    run_id: str
    stores: dict[str, PgVectorRagStore]
    owned: list[tuple[PgVectorRagStore, VectorStoreScope, str]] = field(
        default_factory=list
    )

    def own(
        self,
        store: PgVectorRagStore,
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
                ids = list(store.list_source_record_ids(source_id=source_id, scope=scope))
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
        for store in self.stores.values():
            store.close()
        if errors:
            pytest.fail("PgVector qualification cleanup failed: " + "; ".join(errors))


@pytest.fixture
def live_context() -> Generator[_LiveContext, None, None]:
    run_id = uuid.uuid4().hex
    context = _LiveContext(
        run_id=run_id,
        stores={
            "tenant_a": _open_live_store(f"r2-{run_id}-tenant-a"),
            "tenant_b": _open_live_store(f"r2-{run_id}-tenant-b"),
        },
    )
    try:
        yield context
    finally:
        context.cleanup()


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
                    "metadata": {},
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
                    strategy_id="rag-live-15a-r2",
                    chunk_index=index,
                )
                for index, content in enumerate(document.content.split("|"))
                if content.strip()
            )
        return chunks


class _QualificationEmbeddingManager(BaseEmbeddingManager):
    dimension = DEFAULT_DIMENSION

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
            embeddings=self.embed_texts([document.content for document in documents_tuple]),
        )


def _build_pipeline(
    store: PgVectorRagStore,
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
    store: VectorstoreManager,
    source_id: str,
    scope: VectorStoreScope,
) -> set[str]:
    return set(store.list_source_record_ids(source_id=source_id, scope=scope))


def test_pgvector_live_qualification(live_context: _LiveContext, tmp_path: Path) -> None:
    store_a = live_context.stores["tenant_a"]
    store_b = live_context.stores["tenant_b"]
    tenant_a = VectorStoreScope(
        tenant_id=store_a._tenant_id,
        namespace=f"r2-{live_context.run_id}-namespace-a",
        workspace_id=f"r2-{live_context.run_id}-workspace-a",
    )
    tenant_a_other_namespace = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=f"r2-{live_context.run_id}-namespace-b",
        workspace_id=tenant_a.workspace_id,
    )
    tenant_a_other_workspace = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=tenant_a.namespace,
        workspace_id=f"r2-{live_context.run_id}-workspace-b",
    )
    tenant_a_combined_reverse = VectorStoreScope(
        tenant_id=tenant_a.tenant_id,
        namespace=tenant_a_other_namespace.namespace,
        workspace_id=tenant_a_other_workspace.workspace_id,
    )
    tenant_b = VectorStoreScope(
        tenant_id=store_b._tenant_id,
        namespace=tenant_a.namespace,
        workspace_id=tenant_a.workspace_id,
    )
    all_scopes = (
        (store_a, tenant_a),
        (store_a, tenant_a_other_namespace),
        (store_a, tenant_a_other_workspace),
        (store_a, tenant_a_combined_reverse),
        (store_b, tenant_b),
    )

    # The exact-scope records deliberately have weaker similarity than foreign
    # records, so unscoped or client-side filtering would select the wrong row.
    matrix_ids: dict[VectorStoreScope, str] = {}
    for index, (store, scope) in enumerate(all_scopes):
        source_id = f"matrix://{live_context.run_id}/{index}"
        vector_id = f"logical://{live_context.run_id}/matrix/{index}"
        matrix_ids[scope] = vector_id
        live_context.own(store, scope, source_id)
        store.add_records(
            [
                _record(
                    vector_id,
                    source_id=source_id,
                    scope=scope,
                    embedding=[0.6, 0.8, 0.0, 0.0],
                    metadata={"group": index % 2},
                ),
                _record(
                    f"{vector_id}/adversarial",
                    source_id=f"{source_id}/foreign",
                    scope=scope,
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    metadata={"group": 99},
                ),
            ],
            scope=scope,
        )
        live_context.own(store, scope, f"{source_id}/foreign")

    for store, scope in all_scopes:
        hits = store.query(
            [1.0, 0.0, 0.0, 0.0],
            scope=scope,
            top_k=20,
        )
        assert {hit.vector_id for hit in hits} == {
            matrix_ids[scope],
            f"{matrix_ids[scope]}/adversarial",
        }
        assert all(
            hit.document.scope.tenant_id == scope.tenant_id
            and hit.document.scope.namespace == scope.namespace
            and hit.document.scope.workspace_id == scope.workspace_id
            for hit in hits
        )
        assert store.count(scope=scope) == 2

    with pytest.raises(VectorStoreContractError):
        store_a.query(
            [1.0, 0.0, 0.0, 0.0],
            scope=tenant_b,
            top_k=5,
        )
    assert "tenant" not in {
        hit.vector_id
        for hit in store_a.query(
            [1.0, 0.0, 0.0, 0.0],
            scope=tenant_a_other_namespace,
            top_k=20,
        )
    }

    identity_source = f"identity://{live_context.run_id}"
    identity_ids = [
        f"logical://{live_context.run_id}/identity/1",
        f"logical://{live_context.run_id}/identity/2",
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
            top_k=10,
        )
        if hit.document.provenance.source_id == identity_source
    }
    owned_ids = tuple(
        store_a.list_source_record_ids(source_id=identity_source, scope=tenant_a)
    )
    delete_ids = tuple(sorted(owned_ids))
    assert set(added_ids) == queried_ids == set(owned_ids) == set(identity_ids)
    assert all(not isinstance(vector_id, int) for vector_id in added_ids)
    store_a.delete(delete_ids, scope=tenant_a)
    assert store_a.list_source_record_ids(
        source_id=identity_source,
        scope=tenant_a,
    ) == []

    metadata_source = f"metadata://{live_context.run_id}"
    live_context.own(store_a, tenant_a, metadata_source)
    metadata_records = [
        _record(
            f"logical://{live_context.run_id}/metadata/{index}",
            source_id=metadata_source,
            scope=tenant_a,
            embedding=[0.9, 0.1, 0.0, 0.0],
            metadata={"group": index},
        )
        for index in range(2)
    ]
    store_a.add_records(metadata_records, scope=tenant_a)
    metadata_manager = VectorstoreManager(store_a, scope=tenant_a)
    filtered_hits = metadata_manager.query(
        [0.9, 0.1, 0.0, 0.0],
        scope=tenant_a,
        top_k=20,
        metadata_filter=MetadataFilter(conditions={"group": 1}),
    )
    assert [hit.document.metadata["group"] for hit in filtered_hits] == [1]
    assert all(
        hit.document.scope.tenant_id == tenant_a.tenant_id
        and hit.document.scope.namespace == tenant_a.namespace
        and hit.document.scope.workspace_id == tenant_a.workspace_id
        for hit in filtered_hits
    )
    with pytest.raises(VectorStoreContractError):
        MetadataFilter(conditions={"tenant_id": "foreign"})

    delete_scopes = (
        tenant_a,
        tenant_a_other_namespace,
        tenant_a_other_workspace,
        tenant_a_combined_reverse,
        tenant_b,
    )
    delete_ids_by_scope: dict[VectorStoreScope, str] = {}
    for index, (store, scope) in enumerate(
        zip((store_a, store_a, store_a, store_a, store_b), delete_scopes)
    ):
        source_id = f"delete://{live_context.run_id}/{index}"
        vector_id = f"logical://{live_context.run_id}/delete/{index}"
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
    current_delete_source = f"delete://{live_context.run_id}/0"
    store_a.delete([delete_ids_by_scope[tenant_a]], scope=tenant_a)
    assert store_a.count(scope=tenant_a) == 4
    assert store_a.list_source_record_ids(
        source_id=current_delete_source,
        scope=tenant_a,
    ) == []
    for index, (store, scope) in enumerate(
        zip((store_a, store_a, store_a, store_b), delete_scopes[1:])
    ):
        assert store.list_source_record_ids(
            source_id=f"delete://{live_context.run_id}/{index + 1}",
            scope=scope,
        ) == [delete_ids_by_scope[scope]]

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
        f"logical://{live_context.run_id}/invalid-dimension",
        source_id=f"failure://{live_context.run_id}/dimension",
        scope=tenant_a,
        embedding=[1.0, 0.0, 0.0],
    )
    live_context.own(
        store_a,
        tenant_a,
        f"failure://{live_context.run_id}/dimension",
    )
    with pytest.raises(IntegrationConfigurationError, match="dimension"):
        store_a.add_records([invalid_dimension], scope=tenant_a)

    manager_a = VectorstoreManager(store_a, scope=tenant_a)
    incompatible = _record(
        f"logical://{live_context.run_id}/incompatible",
        source_id=f"failure://{live_context.run_id}/contract",
        scope=tenant_b,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    live_context.own(
        store_a,
        tenant_a,
        f"failure://{live_context.run_id}/contract",
    )
    with pytest.raises(VectorStoreContractError):
        manager_a.add_records([incompatible], scope=tenant_a)

    failed_store = _open_live_store(f"r2-{live_context.run_id}-failure")
    try:
        with failed_store._connection.cursor() as cursor:
            cursor.execute("SET search_path TO pg_temp")
        failure_source = f"failure://{live_context.run_id}/sql"
        with pytest.raises(Exception) as sql_failure:
            failed_store.list_source_record_ids(
                source_id=failure_source,
                scope=VectorStoreScope(tenant_id=failed_store._tenant_id),
            )
        assert type(sql_failure.value).__module__.startswith(("psycopg", "psycopg2"))
    finally:
        failed_store.close()

    soak_source_ids = [
        f"soak-doc-{index}"
        for index in range(50)
    ]
    soak_scope = VectorStoreScope(tenant_id=store_a._tenant_id)
    for source_id in soak_source_ids:
        live_context.own(store_a, soak_scope, source_id)
    soak = run_vectorstore_soak(
        store_a,
        slug="pgvector",
        config=SoakConfig(
            document_count=50,
            query_rounds=5,
            embedding_dim=DEFAULT_DIMENSION,
            max_p95_query_ms=P95_THRESHOLD_MS,
        ),
    )
    assert soak.passed is True, soak.reason
    assert soak.documents_indexed == 50
    assert soak.queries_executed == 5
    assert soak.p95_query_ms <= P95_THRESHOLD_MS

    print(
        "PGVECTOR_R2 "
        f"run={live_context.run_id} "
        "scope_matrix=PASS tenant=PASS namespace=PASS workspace=PASS combined=PASS "
        "adversarial=PASS identity=PASS ownership=PASS replacement=PASS "
        "same_basename=PASS delete=PASS metadata=PASS "
        f"soak_records={soak.documents_indexed} soak_rounds={soak.queries_executed} "
        f"soak_p95_ms={soak.p95_query_ms:.2f} threshold_ms={P95_THRESHOLD_MS:.2f}"
    )
