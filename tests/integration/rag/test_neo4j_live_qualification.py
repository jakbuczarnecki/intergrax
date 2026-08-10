from __future__ import annotations

import inspect
import os
import time
import uuid
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations._shared.p3.factories import (
    create_neo4j_graph_store as create_neo4j_legacy_graph_store,
)
from intergrax.integrations.providers.graph_store.neo4j.bundle import (
    create_neo4j_graph_store,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.distributed.source_operation import (
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
)
from intergrax.rag.document_splitters.chunk_document import build_derived_chunk
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.contracts.graph_store import GraphScope
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.ingest.ingest_pipeline import IngestPipeline, IngestRequest
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

RUN_ENV = "INTERGRAX_RUN_NEO4J_LIVE"
P95_OBSERVATION_ONLY = True


def _scope_key(scope: VectorStoreScope) -> str:
    return GraphScope.from_object(scope).key


@dataclass
class _Runtime:
    scope: VectorStoreScope
    graph: Neo4jRagGraphStore
    vectorstore: VectorstoreManager
    pipeline: IngestPipeline
    retrieval: RetrievalService


@dataclass
class _LiveContext:
    run_id: str
    integration: Any
    vectorstores: dict[str, InMemoryVectorStore]
    runtimes: dict[VectorStoreScope, _Runtime] = field(default_factory=dict)
    owned: list[tuple[_Runtime, str]] = field(default_factory=list)

    def runtime(self, scope: VectorStoreScope) -> _Runtime:
        if scope in self.runtimes:
            return self.runtimes[scope]
        profile = RagProfile(
            retriever_id="graph_rag",
            fast_retriever_id="graph_rag",
            deep_retriever_id="graph_rag",
            graph_rag_enabled=True,
            graph_store_backend="neo4j",
            enable_rerank=False,
            route_mode="off",
            native_hybrid_enabled=False,
        )
        graph = create_rag_graph_store(
            profile=profile,
            integration_graph_store=self.integration,
            tenant_id=scope.tenant_id,
        )
        assert isinstance(graph, Neo4jRagGraphStore)
        graph.bind_scope(GraphScope.from_object(scope))

        provider = self.vectorstores.setdefault(
            scope.tenant_id,
            InMemoryVectorStore(tenant_id=scope.tenant_id),
        )
        vectorstore = VectorstoreManager(provider, scope=scope)
        embedding = _QualificationEmbeddingManager()
        pipeline = IngestPipeline(
            loader=_QualificationLoader(scope, self.run_id),
            splitter=_QualificationSplitter(),
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
                discover_entry_points=False,
            ),
            profile=profile,
        )
        runtime = _Runtime(scope, graph, vectorstore, pipeline, retrieval)
        self.runtimes[scope] = runtime
        return runtime

    def own(self, runtime: _Runtime, source_id: str) -> None:
        item = (runtime, source_id)
        if item not in self.owned:
            self.owned.append(item)

    def cleanup(self) -> None:
        errors: list[str] = []
        for runtime, source_id in self.owned:
            try:
                runtime.graph.unlink_source(
                    source_id,
                    scope=GraphScope.from_object(runtime.scope),
                )
                ids = list(
                    runtime.vectorstore.list_source_record_ids(
                        source_id=source_id,
                        scope=runtime.scope,
                    )
                )
                if ids:
                    runtime.vectorstore.delete(ids, scope=runtime.scope)
                if runtime.graph._run(  # noqa: SLF001 - qualification inspection
                    "MATCH (e:RagEvidence {scope_key: $scope_key, source_id: $source_id}) "
                    "RETURN count(e) AS remaining",
                    {
                        "scope_key": _scope_key(runtime.scope),
                        "source_id": source_id,
                    },
                )[0]["remaining"] != 0:
                    errors.append(f"{source_id}: graph evidence remains")
                if runtime.vectorstore.list_source_record_ids(
                    source_id=source_id,
                    scope=runtime.scope,
                ):
                    errors.append(f"{source_id}: vector records remain")
            except Exception as exc:  # noqa: BLE001 - cleanup must be reported
                errors.append(f"{source_id}: {type(exc).__name__}")

        for scope in self.runtimes:
            try:
                rows = self.integration.run_query(
                    "MATCH (n) WHERE n.scope_key = $scope_key "
                    "RETURN count(n) AS remaining",
                    parameters={"scope_key": _scope_key(scope)},
                ).records
                if not rows or dict(rows[0])["remaining"] != 0:
                    errors.append(f"{_scope_key(scope)}: scoped graph objects remain")
            except Exception as exc:  # noqa: BLE001 - cleanup must be reported
                errors.append(f"{_scope_key(scope)}: {type(exc).__name__}")

        try:
            self.integration.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must be reported
            errors.append(f"close: {type(exc).__name__}")
        if errors:
            pytest.fail("Neo4j qualification cleanup failed: " + "; ".join(errors))


@pytest.fixture
def live_context() -> Generator[_LiveContext, None, None]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the Neo4j live qualification")

    run_id = uuid.uuid4().hex
    try:
        import neo4j

        assert neo4j.__version__ == "5.28.4"
        integration = create_neo4j_graph_store(
            base_url=os.environ.get("INTERGRAX_NEO4J_URL", "bolt://localhost:7687"),
            user=os.environ.get("INTERGRAX_NEO4J_USER", "neo4j"),
            password=os.environ.get("INTERGRAX_NEO4J_PASSWORD", "intergrax"),
        )
    except (ImportError, IntegrationConfigurationError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"Neo4j backend unavailable during open: {type(exc).__name__}")

    # The provider is open now: this is a runtime assertion, never a skip.
    assert type(integration).__name__ == "Neo4jGraphStoreIntegration"
    result = integration.run_query("RETURN 1 AS value")
    assert result.records and dict(result.records[0])["value"] == 1
    context = _LiveContext(run_id, integration, {})
    try:
        yield context
    finally:
        context.cleanup()


class _QualificationLoader:
    def __init__(self, scope: VectorStoreScope, run_id: str) -> None:
        self._scope = scope
        self._run_id = run_id

    def load_document(self, source: str, **kwargs: object) -> list[KnowledgeDocument]:
        del kwargs
        source_path = Path(source).resolve()
        content = source_path.read_text(encoding="utf-8")
        version = "v3" if "A-v3" in content else "v2" if "A-v2" in content else "v1"
        generation = f"r2:{self._run_id}:{source_path}:{version}"
        document_id = f"document:{self._run_id}:{source_path}"
        return [
            KnowledgeDocument.model_validate(
                {
                    "schema_version": 1,
                    "identity": {
                        "document_id": document_id,
                        "root_document_id": document_id,
                    },
                    "scope": {
                        "tenant_id": self._scope.tenant_id,
                        "namespace": self._scope.namespace,
                        "workspace_id": self._scope.workspace_id,
                    },
                    "content": content,
                    "metadata": {
                        "file_name": source_path.name,
                        SOURCE_PUBLICATION_GENERATION_METADATA_KEY: generation,
                    },
                    "provenance": {
                        "source_kind": "rag-live-15c-r2",
                        "source_id": str(source_path),
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
                    strategy_id="rag-live-15c-r2",
                    chunk_index=index,
                )
                for index, content in enumerate(document.content.split("|"))
                if content.strip()
            )
        return chunks


class _QualificationEmbeddingManager(BaseEmbeddingManager):
    dimension = 4

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for index, text in enumerate(texts):
            if "A-v2" in text or "A-v3" in text or "Quasar Node" in text:
                vectors[index, 2] = 1.0
            elif "Alpha Node" in text and "Beta Node" in text:
                vectors[index, 1] = 1.0
            else:
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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _ingest(context: _LiveContext, runtime: _Runtime, path: Path) -> set[str]:
    source_id = str(path.resolve())
    context.own(runtime, source_id)
    result = runtime.pipeline.run(
        IngestRequest(
            source_path=source_id,
            base_metadata={
                "tenant_id": runtime.scope.tenant_id,
                "namespace": runtime.scope.namespace,
            },
            workspace_id=runtime.scope.workspace_id,
        )
    )
    assert result.used is True
    assert result.reason == "ok"
    owned = set(
        runtime.vectorstore.list_source_record_ids(
            source_id=source_id,
            scope=runtime.scope,
        )
    )
    assert owned == set(result.vector_ids)
    return owned


def _raw(context: _LiveContext, statement: str, **parameters: object) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in context.integration.run_query(
            statement,
            parameters=parameters,
        ).records
    ]


def _retrieve(runtime: _Runtime, query: str):
    return runtime.retrieval.retrieve(
        RetrievalRequest(
            query=query,
            scope=runtime.scope,
            final_top_k=20,
            prefetch_k=20,
            route_tier_override="standard",
        )
    )


def test_neo4j_live_graphrag_baseline(
    live_context: _LiveContext,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = live_context
    run_id = context.run_id
    scope_matrix = (
        VectorStoreScope(
            tenant_id=f"r2-{run_id}-tenant-a",
            namespace=f"r2-{run_id}-namespace-a",
            workspace_id=f"r2-{run_id}-workspace-a",
        ),
        VectorStoreScope(
            tenant_id=f"r2-{run_id}-tenant-a",
            namespace=f"r2-{run_id}-namespace-b",
            workspace_id=f"r2-{run_id}-workspace-a",
        ),
        VectorStoreScope(
            tenant_id=f"r2-{run_id}-tenant-a",
            namespace=f"r2-{run_id}-namespace-a",
            workspace_id=f"r2-{run_id}-workspace-b",
        ),
        VectorStoreScope(
            tenant_id=f"r2-{run_id}-tenant-a",
            namespace=f"r2-{run_id}-namespace-b",
            workspace_id=f"r2-{run_id}-workspace-b",
        ),
        VectorStoreScope(
            tenant_id=f"r2-{run_id}-tenant-b",
            namespace=f"r2-{run_id}-namespace-a",
            workspace_id=f"r2-{run_id}-workspace-a",
        ),
    )
    matrix_ids: dict[VectorStoreScope, set[str]] = {}
    for index, scope in enumerate(scope_matrix):
        runtime = context.runtime(scope)
        path = tmp_path / "matrix" / str(index) / "same.txt"
        _write(path, "Matrix Alpha Node connects Matrix Beta Node.")
        matrix_ids[scope] = _ingest(context, runtime, path)

    matrix_entity_id = "ent:matrix_alpha_node"
    physical = _raw(
        context,
        "MATCH (n:RagEntity {id: $id}) "
        "RETURN collect(n.scope_key) AS scope_keys, count(n) AS count",
        id=matrix_entity_id,
    )[0]
    assert physical["count"] == len(scope_matrix)
    assert set(physical["scope_keys"]) == {_scope_key(scope) for scope in scope_matrix}
    for scope in scope_matrix:
        runtime = context.runtime(scope)
        found = runtime.graph.find_nodes(label_contains="Matrix", limit=20)
        assert {node.id for node in found} == {
            matrix_entity_id,
            "ent:matrix_beta_node",
        }
        assert runtime.graph.node_ids_for_chunks(matrix_ids[scope])
        assert _retrieve(runtime, "Matrix Alpha Node").used is True
        assert all(
            chunk.scope.get("tenant_id") == scope.tenant_id
            and chunk.scope.get("namespace") == scope.namespace
            and chunk.scope.get("workspace_id") == scope.workspace_id
            for chunk in _retrieve(runtime, "Matrix Alpha Node").chunks
        )

    chunk_rows = _raw(
        context,
        "MATCH (c:RagChunk {scope_key: $scope_key}) "
        "RETURN c.scope_key AS scope_key, c.id AS id",
        scope_key=_scope_key(scope_matrix[0]),
    )
    assert chunk_rows and all(
        row["scope_key"] == _scope_key(scope_matrix[0]) for row in chunk_rows
    )

    scope = scope_matrix[0]
    runtime = context.runtime(scope)
    source_a = tmp_path / "a" / "same.txt"
    source_b = tmp_path / "b" / "same.txt"
    source_a_id = str(source_a.resolve())
    source_b_id = str(source_b.resolve())
    _write(source_a, "A-v1 Alpha Node connects Beta Node|A-v1 Beta Node connects Gamma Node.")
    _write(source_b, "B-stable Alpha Node connects Beta Node.")
    a_v1 = _ingest(context, runtime, source_a)
    b_v1 = _ingest(context, runtime, source_b)
    assert a_v1 and b_v1 and a_v1.isdisjoint(b_v1)
    assert source_a.name == source_b.name and source_a_id != source_b_id

    evidence = _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key}) "
        "RETURN collect(DISTINCT e.source_id) AS source_ids, "
        "count(e) AS count",
        scope_key=_scope_key(scope),
    )[0]
    assert source_a_id in evidence["source_ids"]
    assert source_b_id in evidence["source_ids"]
    assert evidence["count"] > 0
    generation = _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key, source_id: $source_id}) "
        "RETURN collect(DISTINCT e.generation) AS generations",
        scope_key=_scope_key(scope),
        source_id=source_a_id,
    )[0]["generations"]
    assert generation and all(value for value in generation)

    shared_edge_key = (
        f"{_scope_key(scope)}|ent:alpha_node|co_occurs|ent:beta_node"
    )
    supporters = _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key, edge_key: $edge_key}) "
        "RETURN collect(DISTINCT e.source_id) AS source_ids, count(e) AS count",
        scope_key=_scope_key(scope),
        edge_key=shared_edge_key,
    )[0]
    assert set(supporters["source_ids"]) == {source_a_id, source_b_id}
    assert supporters["count"] == 2

    assert runtime.graph.unlink_source(
        source_a_id,
        scope=GraphScope.from_object(scope),
    ) > 0
    a_vector_ids = list(
        runtime.vectorstore.list_source_record_ids(
            source_id=source_a_id,
            scope=scope,
        )
    )
    runtime.vectorstore.delete(a_vector_ids, scope=scope)
    assert runtime.graph.node_ids_for_chunks(b_v1)
    assert "ent:beta_node" in {node.id for node in runtime.graph.neighbors("ent:alpha_node")}
    supporters_after_a = _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key, edge_key: $edge_key}) "
        "RETURN collect(DISTINCT e.source_id) AS source_ids",
        scope_key=_scope_key(scope),
        edge_key=shared_edge_key,
    )[0]["source_ids"]
    assert supporters_after_a == [source_b_id]

    _write(source_a, "A-v1 Alpha Node connects Beta Node|A-v1 Beta Node connects Gamma Node.")
    a_v1 = _ingest(context, runtime, source_a)
    _write(source_a, "A-v2 Alpha Node connects Quasar Node|A-v2 Quasar Node connects Zeta Node.")
    a_v2 = _ingest(context, runtime, source_a)
    assert a_v1.isdisjoint(a_v2)
    assert runtime.graph.node_ids_for_chunks(a_v1) == set()
    old_a_only_edge = (
        f"{_scope_key(scope)}|ent:beta_node|co_occurs|ent:gamma_node"
    )
    assert _raw(
        context,
        "MATCH ()-[r:RAG_REL {scope_key: $scope_key, edge_key: $edge_key}]->() "
        "RETURN count(r) AS count",
        scope_key=_scope_key(scope),
        edge_key=old_a_only_edge,
    )[0]["count"] == 0
    assert runtime.graph.node_ids_for_chunks(b_v1)
    assert _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key, edge_key: $edge_key}) "
        "RETURN collect(DISTINCT e.source_id) AS source_ids",
        scope_key=_scope_key(scope),
        edge_key=shared_edge_key,
    )[0]["source_ids"] == [source_b_id]

    _write(source_a, "A-v3 Alpha Node connects Quasar Node.")
    a_v3 = _ingest(context, runtime, source_a)
    assert a_v2.isdisjoint(a_v3)
    assert runtime.graph.node_ids_for_chunks(a_v2) == set()
    assert "ent:zeta_node" not in {node.id for node in runtime.graph.find_nodes(label_contains="Zeta")}
    repeated_v3 = _ingest(context, runtime, source_a)
    assert repeated_v3 == a_v3
    assert runtime.graph.node_ids_for_chunks(b_v1)

    current = _retrieve(runtime, "A-v3 Alpha Node Quasar Node")
    assert current.used is True
    assert any("A-v3" in chunk.text for chunk in current.chunks)
    assert all("A-v2" not in chunk.text and "A-v1" not in chunk.text for chunk in current.chunks)
    assert current.trace.graph_expanded_node_ids
    shared = _retrieve(runtime, "Alpha Node Beta Node")
    assert shared.used is True
    assert any("B-stable" in chunk.text for chunk in shared.chunks)
    assert all(
        chunk.scope.get("tenant_id") == scope.tenant_id
        and chunk.scope.get("namespace") == scope.namespace
        and chunk.scope.get("workspace_id") == scope.workspace_id
        for chunk in shared.chunks
    )

    assert runtime.graph.unlink_source(
        source_a_id,
        scope=GraphScope.from_object(scope),
    ) > 0
    runtime.vectorstore.delete(
        list(
            runtime.vectorstore.list_source_record_ids(
                source_id=source_a_id,
                scope=scope,
            )
        ),
        scope=scope,
    )
    assert runtime.graph.node_ids_for_chunks(b_v1)
    assert set(
        runtime.vectorstore.list_source_record_ids(
            source_id=source_b_id,
            scope=scope,
        )
    ) == b_v1
    for foreign_scope in scope_matrix[1:]:
        foreign_runtime = context.runtime(foreign_scope)
        assert foreign_runtime.graph.find_nodes(label_contains="Matrix", limit=20)

    assert runtime.graph.unlink_source(
        source_b_id,
        scope=GraphScope.from_object(scope),
    ) > 0
    runtime.vectorstore.delete(
        list(
            runtime.vectorstore.list_source_record_ids(
                source_id=source_b_id,
                scope=scope,
            )
        ),
        scope=scope,
    )
    assert _raw(
        context,
        "MATCH ()-[r:RAG_REL {scope_key: $scope_key, edge_key: $edge_key}]->() "
        "RETURN count(r) AS count",
        scope_key=_scope_key(scope),
        edge_key=shared_edge_key,
    )[0]["count"] == 0

    with pytest.raises(ValueError, match="graph scope cannot change"):
        runtime.graph.bind_scope(
            GraphScope(
                scope.tenant_id,
                namespace="foreign-namespace",
                workspace_id=scope.workspace_id,
            )
        )
    integration_type = type(context.integration)
    with monkeypatch.context() as patch:
        patch.setattr(
            integration_type,
            "run_query",
            lambda self, *args, **kwargs: object(),
        )
        with pytest.raises(RuntimeError, match="malformed result"):
            runtime.graph.find_nodes(label_contains="Matrix")
    with monkeypatch.context() as patch:
        patch.setattr(
            integration_type,
            "run_query",
            lambda self, *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("cypher error")
            ),
        )
        with pytest.raises(RuntimeError, match="cypher error"):
            runtime.graph.find_nodes(label_contains="Matrix")
        with pytest.raises(RuntimeError, match="cypher error"):
            runtime.graph.unlink_source("ownership://failure", scope=GraphScope.from_object(scope))
        with pytest.raises(RuntimeError, match="cypher error"):
            runtime.graph.neighbors("ent:matrix_alpha_node")
    assert runtime.graph.unlink_source(
        "missing://zero-row",
        scope=GraphScope.from_object(scope),
    ) == 0
    factory_source = inspect.getsource(create_neo4j_legacy_graph_store)
    assert "execute_write" in factory_source
    assert "session" in factory_source

    soak_scope = VectorStoreScope(
        tenant_id=scope.tenant_id,
        namespace=f"r2-{run_id}-soak-namespace",
        workspace_id=f"r2-{run_id}-soak-workspace",
    )
    soak_runtime = context.runtime(soak_scope)
    soak_paths = []
    for index in range(16):
        path = tmp_path / "soak" / f"source-{index}" / "same.txt"
        _write(path, "Soak Alpha Node connects Soak Beta Node.")
        soak_paths.append(path)
        _ingest(context, soak_runtime, path)
    soak_evidence = _raw(
        context,
        "MATCH (e:RagEvidence {scope_key: $scope_key}) RETURN count(e) AS count",
        scope_key=_scope_key(soak_scope),
    )[0]["count"]
    assert 30 <= soak_evidence <= 50
    latencies: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        assert "ent:soak_beta_node" in {
            node.id
            for node in soak_runtime.graph.neighbors("ent:soak_alpha_node")
        }
        latencies.append((time.perf_counter() - started) * 1000.0)
    p95_ms = sorted(latencies)[-1]
    assert P95_OBSERVATION_ONLY is True
    print(
        f"NEO4J_R2 run_id={run_id} scope_matrix=5 "
        f"soak_evidence={soak_evidence} traversal_rounds=5 p95_ms={p95_ms:.2f}"
    )
