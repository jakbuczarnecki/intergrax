from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from intergrax.distributed.source_operation import (
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    SourceOperationLease,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.neo4j.bundle import (
    create_neo4j_graph_store,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphScope
from intergrax.rag.graph.generation_visibility import cypher_evidence_visible
from intergrax.rag.graph.providers.cypher_rag_graph_store import CypherRagGraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.integration, pytest.mark.gate]

RUN_ENV = "INTERGRAX_RUN_NEO4J_LIVE"
CONTENTION_ITERATIONS = 20


class _FailingCoordinator:
    def active_publication_generation(
        self, *, key: RagSourceOperationKey
    ) -> str | None:
        del key
        raise RuntimeError("coordinator unavailable")


@dataclass(frozen=True)
class _Harness:
    run_id: str
    integration: Any
    graph: Neo4jRagGraphStore | InMemoryGraphStore
    coordinator: InProcessSourceOperationCoordinator
    scope: GraphScope

    @property
    def source_prefix(self) -> str:
        return f"r2d2-{self.run_id}"


def _metadata(
    harness: _Harness,
    *,
    source_id: str,
    generation: str,
    chunk_id: str,
) -> dict[str, object]:
    return {
        "tenant_id": harness.scope.tenant_id,
        "namespace": harness.scope.namespace,
        "workspace_id": harness.scope.workspace_id,
        "source_id": source_id,
        "chunk_ids": [chunk_id],
        SOURCE_PUBLICATION_GENERATION_METADATA_KEY: generation,
    }


def _source_key(harness: _Harness, source_id: str) -> RagSourceOperationKey:
    return RagSourceOperationKey(
        tenant_id=harness.scope.tenant_id,
        namespace=harness.scope.namespace,
        workspace_id=harness.scope.workspace_id,
        source_id=source_id,
    )


def _promote_generation(
    harness: _Harness,
    *,
    source_id: str,
) -> str:
    lease = harness.coordinator.acquire(key=_source_key(harness, source_id))
    assert lease is not None
    generation = harness.coordinator.publication_generation(lease=lease)
    assert harness.coordinator.promote_publication(lease=lease)
    harness.coordinator.release(lease=lease)
    return generation


def _acquire_generation(
    harness: _Harness,
    *,
    source_id: str,
    promote: bool,
) -> tuple[str, SourceOperationLease]:
    lease = harness.coordinator.acquire(key=_source_key(harness, source_id))
    assert lease is not None
    generation = harness.coordinator.publication_generation(lease=lease)
    if promote:
        assert harness.coordinator.promote_publication(lease=lease)
        harness.coordinator.release(lease=lease)
    return generation, lease


def _write_chain(
    harness: _Harness,
    *,
    source_id: str,
    generation: str,
    node_ids: Sequence[str],
    chunk_prefix: str,
) -> list[str]:
    chunk_ids: list[str] = []
    for index, node_id in enumerate(node_ids):
        chunk_id = f"{chunk_prefix}-{index}"
        chunk_ids.append(chunk_id)
        metadata = _metadata(
            harness,
            source_id=source_id,
            generation=generation,
            chunk_id=chunk_id,
        )
        harness.graph.upsert_node(GraphNode(node_id, node_id, metadata=metadata))
        harness.graph.link_chunk(node_id, chunk_id)
        if index > 0:
            harness.graph.upsert_edge(
                GraphEdge(
                    node_ids[index - 1],
                    node_id,
                    "supports",
                    metadata=metadata,
                )
            )
    return chunk_ids


def _write_edge(
    harness: _Harness,
    *,
    source_id: str,
    generation: str,
    from_id: str,
    to_id: str,
    chunk_id: str,
) -> None:
    metadata = _metadata(
        harness,
        source_id=source_id,
        generation=generation,
        chunk_id=chunk_id,
    )
    harness.graph.upsert_node(GraphNode(from_id, from_id, metadata=metadata))
    harness.graph.upsert_node(GraphNode(to_id, to_id, metadata=metadata))
    harness.graph.link_chunk(from_id, chunk_id)
    harness.graph.upsert_edge(GraphEdge(from_id, to_id, "supports", metadata=metadata))


def _neighbor_ids(
    graph: Neo4jRagGraphStore | InMemoryGraphStore, node_id: str
) -> set[str]:
    return {node.id for node in graph.neighbors(node_id)}


def _path_visible(
    graph: Neo4jRagGraphStore | InMemoryGraphStore,
    *node_ids: str,
) -> bool:
    if len(node_ids) < 2:
        return True
    current = node_ids[0]
    for nxt in node_ids[1:]:
        if nxt not in _neighbor_ids(graph, current):
            return False
        current = nxt
    return True


def _physical_evidence_count(
    graph: Neo4jRagGraphStore | InMemoryGraphStore,
    *,
    source_id: str,
    generation: str,
) -> int:
    scope = graph.scope
    assert scope is not None
    if isinstance(graph, CypherRagGraphStore):
        rows = graph._run(  # noqa: SLF001 - qualification proof query
            """
            MATCH (e:RagEvidence {
                scope_key: $scope_key,
                source_id: $source_id,
                generation: $generation
            })
            RETURN count(e) AS count
            """,
            {
                "scope_key": scope.key,
                "source_id": source_id,
                "generation": generation,
            },
        )
        return int(rows[0]["count"])
    count = 0
    for evidence_sets in (
        graph._node_evidence.values(),  # noqa: SLF001 - in-memory physical proof
        graph._edge_evidence.values(),
    ):
        for evidence_set in evidence_sets:
            for evidence in evidence_set:
                if (
                    evidence.source_id == source_id
                    and evidence.generation == generation
                ):
                    count += 1
    return count


def _assert_server_side_fencing(graph: Neo4jRagGraphStore) -> None:
    from intergrax.rag.graph.generation_visibility import visibility_query_params

    predicate = cypher_evidence_visible(alias="e")
    visibility_source = inspect.getsource(visibility_query_params)
    find_source = inspect.getsource(CypherRagGraphStore.find_nodes)
    neighbor_source = inspect.getsource(CypherRagGraphStore.neighbors)
    chunk_source = inspect.getsource(CypherRagGraphStore.chunk_ids_for_nodes)
    for source in (visibility_source, find_source, neighbor_source, chunk_source):
        assert "coordinator_bound" in source or "coordinator_bound" in predicate
        assert "active_pairs" in source or "active_pairs" in predicate
    assert "_visibility_params" in find_source

    captured: list[tuple[str, dict[str, object]]] = []
    original_run = graph._run  # noqa: SLF001 - qualification seam

    def capture_run(
        statement: str, parameters: dict[str, object]
    ) -> list[dict[str, Any]]:
        captured.append((statement, dict(parameters)))
        return original_run(statement, parameters)

    graph._run = capture_run  # type: ignore[method-assign]
    try:
        graph.find_nodes(label_contains="probe", limit=1)
    finally:
        graph._run = original_run  # type: ignore[method-assign]

    visibility_queries = [
        (statement, params)
        for statement, params in captured
        if "pair IN $active_pairs" in statement or "coordinator_bound" in statement
    ]
    assert visibility_queries
    assert any(
        params.get("coordinator_bound") is True and bool(params.get("active_pairs"))
        for _, params in visibility_queries
    )


def _assert_graph_rag_retrieval_fenced(
    harness: _Harness,
    *,
    source_id: str,
    generation: str,
    visible_chunk: str,
    visible_node: str,
    stale_node: str,
) -> None:
    scope = VectorStoreScope(
        tenant_id=harness.scope.tenant_id,
        namespace=harness.scope.namespace,
        workspace_id=harness.scope.workspace_id,
    )
    vector = InMemoryVectorStore(tenant_id=scope.tenant_id)
    manager = VectorstoreManager(vector, scope=scope)
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": f"document-{visible_chunk}",
                "root_document_id": f"document-{visible_chunk}",
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": f"Qualification seed for {visible_node}",
            "metadata": {},
            "provenance": {
                "source_kind": "rag-live-15d-r2",
                "source_id": source_id,
            },
        }
    )
    manager.add_documents([document], [[1.0, 0.0, 0.0]], ids=[visible_chunk])

    class _Emb(BaseEmbeddingManager):
        dimension = 3

        def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
            return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)

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

    retriever = GraphRagRetriever(
        manager,
        _Emb(),
        harness.graph,
        source_coordinator=harness.coordinator,
    )
    hits = retriever.retrieve(
        RetrieverQuery(
            query_text=visible_node,
            query_embedding=None,
            top_k=5,
            scope=scope,
        )
    )
    trace = retriever.last_graph_trace
    assert hits
    assert trace is not None
    assert visible_node in trace.expanded_node_ids
    assert stale_node not in trace.expanded_node_ids


def _open_neo4j_harness(run_id: str) -> _Harness:
    integration = create_neo4j_graph_store(
        base_url=os.environ.get("INTERGRAX_NEO4J_URL", "bolt://localhost:7687"),
        user=os.environ.get("INTERGRAX_NEO4J_USER", "neo4j"),
        password=os.environ.get("INTERGRAX_NEO4J_PASSWORD", "intergrax"),
    )
    assert type(integration).__name__ == "Neo4jGraphStoreIntegration"
    result = integration.run_query("RETURN 1 AS value")
    assert result.records and dict(result.records[0])["value"] == 1

    scope = GraphScope(
        f"r2d2-{run_id}-tenant",
        namespace=f"r2d2-{run_id}-namespace",
        workspace_id=f"r2d2-{run_id}-workspace",
    )
    profile = RagProfile(graph_store_backend="neo4j", graph_rag_enabled=True)
    graph = create_rag_graph_store(
        profile=profile,
        integration_graph_store=integration,
        tenant_id=scope.tenant_id,
    )
    assert isinstance(graph, Neo4jRagGraphStore)
    graph.bind_scope(scope)
    coordinator = InProcessSourceOperationCoordinator(
        owner_id=f"r2d2-{run_id}",
        token_factory=lambda: f"token-{run_id}",
    )
    graph.set_source_operation_coordinator(coordinator)
    return _Harness(run_id, integration, graph, coordinator, scope)


def _purge_scope(graph: Neo4jRagGraphStore | InMemoryGraphStore) -> None:
    graph.purge_graph()


def _cleanup_harness(
    harness: _Harness, *, extra_scopes: Sequence[GraphScope] = ()
) -> None:
    errors: list[str] = []
    scopes = [harness.scope]
    try:
        harness.graph.purge_graph()
    except Exception as exc:  # noqa: BLE001 - cleanup must be reported
        errors.append(f"purge: {type(exc).__name__}")
    if isinstance(harness.graph, Neo4jRagGraphStore):
        for scope in scopes:
            try:
                harness.integration.run_query(
                    "MATCH (n) WHERE n.scope_key = $scope_key DETACH DELETE n",
                    parameters={"scope_key": scope.key},
                )
                rows = harness.integration.run_query(
                    "MATCH (n) WHERE n.scope_key = $scope_key RETURN count(n) AS remaining",
                    parameters={"scope_key": scope.key},
                ).records
                if not rows or dict(rows[0])["remaining"] != 0:
                    errors.append(f"{scope.key}: scoped graph objects remain")
            except Exception as exc:  # noqa: BLE001 - cleanup must be reported
                errors.append(f"verify: {type(exc).__name__}")
        try:
            harness.integration.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must be reported
            errors.append(f"close: {type(exc).__name__}")
    if errors:
        pytest.fail("Neo4j generation fencing cleanup failed: " + "; ".join(errors))


def _run_qualification_phases(harness: _Harness) -> list[GraphScope]:
    graph = harness.graph
    prefix = harness.source_prefix
    source = f"{prefix}-takeover"
    extra_scopes: list[GraphScope] = []

    g1 = _promote_generation(harness, source_id=source)
    _write_chain(
        harness,
        source_id=source,
        generation=g1,
        node_ids=[f"{prefix}:alpha", f"{prefix}:beta", f"{prefix}:gamma"],
        chunk_prefix=f"{prefix}-g1",
    )
    assert graph.find_nodes(label_contains=f"{prefix}:beta", limit=5)
    assert _path_visible(graph, f"{prefix}:alpha", f"{prefix}:beta", f"{prefix}:gamma")

    g2 = _promote_generation(harness, source_id=source)
    _write_chain(
        harness,
        source_id=source,
        generation=g2,
        node_ids=[f"{prefix}:alpha", f"{prefix}:quasar", f"{prefix}:zeta"],
        chunk_prefix=f"{prefix}-g2",
    )
    assert graph.find_nodes(label_contains=f"{prefix}:beta", limit=5) == []
    assert graph.find_nodes(label_contains=f"{prefix}:quasar", limit=5)
    assert _path_visible(graph, f"{prefix}:alpha", f"{prefix}:quasar", f"{prefix}:zeta")
    assert not _path_visible(
        graph, f"{prefix}:alpha", f"{prefix}:beta", f"{prefix}:gamma"
    )

    _write_edge(
        harness,
        source_id=source,
        generation=g1,
        from_id=f"{prefix}:late-g1",
        to_id=f"{prefix}:quasar",
        chunk_id=f"{prefix}-late-g1",
    )
    assert _physical_evidence_count(graph, source_id=source, generation=g1) > 0
    assert graph.find_nodes(label_contains=f"{prefix}:late-g1", limit=5) == []
    assert graph.node_ids_for_chunks({f"{prefix}-late-g1"}) == set()
    assert graph.find_nodes(label_contains=f"{prefix}:quasar", limit=5)

    reverse_source = f"{prefix}-reverse-a"
    g1_rev, lease_g1 = _acquire_generation(
        harness,
        source_id=reverse_source,
        promote=False,
    )
    _write_chain(
        harness,
        source_id=reverse_source,
        generation=g1_rev,
        node_ids=[f"{prefix}:rev-a", f"{prefix}:rev-b"],
        chunk_prefix=f"{prefix}-rev-g1",
    )
    harness.coordinator.release(lease=lease_g1)
    g2_rev, lease_g2 = _acquire_generation(
        harness,
        source_id=reverse_source,
        promote=False,
    )
    assert harness.coordinator.promote_publication(lease=lease_g2)
    _write_chain(
        harness,
        source_id=reverse_source,
        generation=g2_rev,
        node_ids=[f"{prefix}:rev-a", f"{prefix}:rev-c"],
        chunk_prefix=f"{prefix}-rev-g2",
    )
    harness.coordinator.release(lease=lease_g2)
    _write_edge(
        harness,
        source_id=reverse_source,
        generation=g1_rev,
        from_id=f"{prefix}:rev-late",
        to_id=f"{prefix}:rev-c",
        chunk_id=f"{prefix}-rev-late",
    )
    assert graph.find_nodes(label_contains=f"{prefix}:rev-c", limit=5)
    assert graph.find_nodes(label_contains=f"{prefix}:rev-b", limit=5) == []
    assert graph.find_nodes(label_contains=f"{prefix}:rev-late", limit=5) == []

    partial_source = f"{prefix}-reverse-b"
    g1_partial, lease_partial = _acquire_generation(
        harness,
        source_id=partial_source,
        promote=False,
    )
    _write_chain(
        harness,
        source_id=partial_source,
        generation=g1_partial,
        node_ids=[f"{prefix}:pb-a", f"{prefix}:pb-b"],
        chunk_prefix=f"{prefix}-pb-g1",
    )
    harness.coordinator.release(lease=lease_partial)
    g2_partial = _promote_generation(harness, source_id=partial_source)
    _write_chain(
        harness,
        source_id=partial_source,
        generation=g2_partial,
        node_ids=[f"{prefix}:pb-a", f"{prefix}:pb-c"],
        chunk_prefix=f"{prefix}-pb-g2",
    )
    assert (
        _physical_evidence_count(
            graph,
            source_id=partial_source,
            generation=g1_partial,
        )
        > 0
    )
    assert graph.find_nodes(label_contains=f"{prefix}:pb-b", limit=5) == []
    assert graph.find_nodes(label_contains=f"{prefix}:pb-c", limit=5)
    removed = graph.unlink_source_generation(
        partial_source,
        g1_partial,
        scope=harness.scope,
    )
    assert removed > 0
    assert graph.find_nodes(label_contains=f"{prefix}:pb-c", limit=5)

    reduced_source = f"{prefix}-reduced"
    g1_reduced = _promote_generation(harness, source_id=reduced_source)
    _write_chain(
        harness,
        source_id=reduced_source,
        generation=g1_reduced,
        node_ids=[
            f"{prefix}:rd-a",
            f"{prefix}:rd-b",
            f"{prefix}:rd-c",
            f"{prefix}:rd-d",
        ],
        chunk_prefix=f"{prefix}-rd-g1",
    )
    g2_reduced = _promote_generation(harness, source_id=reduced_source)
    _write_chain(
        harness,
        source_id=reduced_source,
        generation=g2_reduced,
        node_ids=[f"{prefix}:rd-a", f"{prefix}:rd-b"],
        chunk_prefix=f"{prefix}-rd-g2",
    )
    assert _path_visible(graph, f"{prefix}:rd-a", f"{prefix}:rd-b")
    assert not _path_visible(graph, f"{prefix}:rd-b", f"{prefix}:rd-c")
    assert graph.find_nodes(label_contains=f"{prefix}:rd-d", limit=5) == []

    removed_g1 = graph.unlink_source_generation(source, g1, scope=harness.scope)
    assert removed_g1 > 0
    assert graph.find_nodes(label_contains=f"{prefix}:quasar", limit=5)
    assert graph.unlink_source_generation(source, g1, scope=harness.scope) == 0

    shared_a = f"{prefix}-shared-a"
    shared_b = f"{prefix}-shared-b"
    g1_shared = _promote_generation(harness, source_id=shared_a)
    _write_edge(
        harness,
        source_id=shared_a,
        generation=g1_shared,
        from_id=f"{prefix}:shared-x",
        to_id=f"{prefix}:shared-y",
        chunk_id=f"{prefix}-shared-a-g1",
    )
    g_b = _promote_generation(harness, source_id=shared_b)
    _write_edge(
        harness,
        source_id=shared_b,
        generation=g_b,
        from_id=f"{prefix}:shared-x",
        to_id=f"{prefix}:shared-y",
        chunk_id=f"{prefix}-shared-b",
    )
    g2_shared = _promote_generation(harness, source_id=shared_a)
    _write_edge(
        harness,
        source_id=shared_a,
        generation=g2_shared,
        from_id=f"{prefix}:shared-x",
        to_id=f"{prefix}:shared-a2",
        chunk_id=f"{prefix}-shared-a-g2",
    )
    assert any(
        node.id == f"{prefix}:shared-y"
        for node in graph.neighbors(f"{prefix}:shared-x")
    )
    assert any(
        node.id == f"{prefix}:shared-a2"
        for node in graph.neighbors(f"{prefix}:shared-x")
    )
    graph.unlink_source_generation(shared_a, g1_shared, scope=harness.scope)
    assert any(
        node.id == f"{prefix}:shared-y"
        for node in graph.neighbors(f"{prefix}:shared-x")
    )
    graph.unlink_source(shared_b, scope=harness.scope)
    assert all(
        node.id != f"{prefix}:shared-y"
        for node in graph.neighbors(f"{prefix}:shared-x")
    )
    assert any(
        node.id == f"{prefix}:shared-a2"
        for node in graph.neighbors(f"{prefix}:shared-x")
    )

    for index, other_scope in enumerate(
        (
            GraphScope(
                harness.scope.tenant_id,
                namespace=f"{prefix}-other-ns",
                workspace_id=harness.scope.workspace_id,
            ),
            GraphScope(
                harness.scope.tenant_id,
                namespace=harness.scope.namespace,
                workspace_id=f"{prefix}-other-ws",
            ),
            GraphScope(
                f"{prefix}-other-tenant",
                namespace=harness.scope.namespace,
                workspace_id=harness.scope.workspace_id,
            ),
        ),
        start=1,
    ):
        extra_scopes.append(other_scope)
        if isinstance(harness.graph, InMemoryGraphStore):
            other_graph: Neo4jRagGraphStore | InMemoryGraphStore = InMemoryGraphStore()
            other_graph.bind_scope(other_scope)
            other_graph.set_source_operation_coordinator(harness.coordinator)
        else:
            other_graph = create_rag_graph_store(
                profile=RagProfile(graph_store_backend="neo4j", graph_rag_enabled=True),
                integration_graph_store=harness.integration,
                tenant_id=other_scope.tenant_id,
            )
            assert isinstance(other_graph, Neo4jRagGraphStore)
            other_graph.bind_scope(other_scope)
            other_graph.set_source_operation_coordinator(harness.coordinator)

        iso_source = f"{prefix}-iso-{index}"
        main_gen = _promote_generation(harness, source_id=f"{prefix}-iso-main")
        other_gen = _promote_generation(harness, source_id=iso_source)
        _write_edge(
            harness,
            source_id=f"{prefix}-iso-main",
            generation=main_gen,
            from_id=f"{prefix}:iso-main",
            to_id=f"{prefix}:iso-main-target",
            chunk_id=f"{prefix}-iso-main",
        )
        other_harness = _Harness(
            harness.run_id,
            harness.integration,
            other_graph,
            harness.coordinator,
            other_scope,
        )
        _write_edge(
            other_harness,
            source_id=iso_source,
            generation=other_gen,
            from_id=f"{prefix}:iso-other",
            to_id=f"{prefix}:iso-other-target",
            chunk_id=f"{prefix}-iso-other-{index}",
        )
        assert graph.find_nodes(label_contains=f"{prefix}:iso-main", limit=5)
        assert other_graph.find_nodes(label_contains=f"{prefix}:iso-other", limit=5)
        assert graph.find_nodes(label_contains=f"{prefix}:iso-other", limit=5) == []
        assert (
            other_graph.find_nodes(label_contains=f"{prefix}:iso-main", limit=5) == []
        )
        _purge_scope(other_graph)

    if isinstance(graph, Neo4jRagGraphStore):
        _assert_server_side_fencing(graph)

    assert graph.find_nodes(label_contains=f"{prefix}:quasar", limit=5)
    assert graph.chunk_ids_for_nodes({f"{prefix}:quasar"}) == [f"{prefix}-g2-1"]
    assert graph.node_ids_for_chunks({f"{prefix}-g2-1"}) == {f"{prefix}:quasar"}
    assert graph.node_ids_for_chunks({f"{prefix}-g1-1"}) == set()

    _assert_graph_rag_retrieval_fenced(
        harness,
        source_id=source,
        generation=g2,
        visible_chunk=f"{prefix}-g2-1",
        visible_node=f"{prefix}:quasar",
        stale_node=f"{prefix}:beta",
    )

    partial_pub_source = f"{prefix}-partial-pub"
    g1_partial_pub, lease_pub = _acquire_generation(
        harness,
        source_id=partial_pub_source,
        promote=False,
    )
    _write_edge(
        harness,
        source_id=partial_pub_source,
        generation=g1_partial_pub,
        from_id=f"{prefix}:pp-a",
        to_id=f"{prefix}:pp-b",
        chunk_id=f"{prefix}-pp-g1",
    )
    harness.coordinator.release(lease=lease_pub)
    g2_partial_pub = _promote_generation(harness, source_id=partial_pub_source)
    _write_edge(
        harness,
        source_id=partial_pub_source,
        generation=g2_partial_pub,
        from_id=f"{prefix}:pp-a",
        to_id=f"{prefix}:pp-c",
        chunk_id=f"{prefix}-pp-g2",
    )
    assert (
        _physical_evidence_count(
            graph,
            source_id=partial_pub_source,
            generation=g1_partial_pub,
        )
        > 0
    )
    assert graph.node_ids_for_chunks({f"{prefix}-pp-g1"}) == set()
    assert not _path_visible(graph, f"{prefix}:pp-a", f"{prefix}:pp-b")
    assert _path_visible(graph, f"{prefix}:pp-a", f"{prefix}:pp-c")
    assert graph.find_nodes(label_contains=f"{prefix}:pp-c", limit=5)

    idem_source = f"{prefix}-idem"
    g2_idem = _promote_generation(harness, source_id=idem_source)
    _write_edge(
        harness,
        source_id=idem_source,
        generation=g2_idem,
        from_id=f"{prefix}:idem",
        to_id=f"{prefix}:idem-target",
        chunk_id=f"{prefix}-idem",
    )
    before = _physical_evidence_count(graph, source_id=idem_source, generation=g2_idem)
    _write_edge(
        harness,
        source_id=idem_source,
        generation=g2_idem,
        from_id=f"{prefix}:idem",
        to_id=f"{prefix}:idem-target",
        chunk_id=f"{prefix}-idem",
    )
    after = _physical_evidence_count(graph, source_id=idem_source, generation=g2_idem)
    assert after == before
    visible_idem = [
        node
        for node in graph.find_nodes(label_contains=f"{prefix}:idem", limit=10)
        if node.id == f"{prefix}:idem"
    ]
    assert len(visible_idem) == 1

    fail_source = f"{prefix}-fail"
    g_fail = _promote_generation(harness, source_id=fail_source)
    _write_edge(
        harness,
        source_id=fail_source,
        generation=g_fail,
        from_id=f"{prefix}:fail",
        to_id=f"{prefix}:fail-target",
        chunk_id=f"{prefix}-fail",
    )
    coordinator = harness.coordinator
    graph.set_source_operation_coordinator(_FailingCoordinator())
    if isinstance(graph, CypherRagGraphStore):
        with pytest.raises(RuntimeError, match="graph generation resolution failed"):
            graph.find_nodes(label_contains=f"{prefix}:fail", limit=5)
        with pytest.raises(RuntimeError, match="graph generation resolution failed"):
            graph.node_ids_for_chunks({f"{prefix}-fail"})
    else:
        visible_fail = [
            node
            for node in graph.find_nodes(label_contains=f"{prefix}:fail", limit=10)
            if node.id == f"{prefix}:fail"
        ]
        assert visible_fail == []
        assert graph.node_ids_for_chunks({f"{prefix}-fail"}) == set()
    graph.set_source_operation_coordinator(coordinator)

    unbound_store = (
        InMemoryGraphStore()
        if isinstance(graph, InMemoryGraphStore)
        else create_rag_graph_store(
            profile=RagProfile(graph_store_backend="neo4j", graph_rag_enabled=True),
            integration_graph_store=harness.integration,
            tenant_id=harness.scope.tenant_id,
        )
    )
    if isinstance(unbound_store, Neo4jRagGraphStore):
        unbound_store.bind_scope(harness.scope)
    else:
        unbound_store.bind_scope(harness.scope)
    unbound_harness = _Harness(
        harness.run_id,
        harness.integration,
        unbound_store,
        harness.coordinator,
        harness.scope,
    )
    _write_edge(
        unbound_harness,
        source_id=f"{prefix}-unbound",
        generation="9:stale",
        from_id=f"{prefix}:versioned",
        to_id=f"{prefix}:versioned-target",
        chunk_id=f"{prefix}-versioned",
    )
    unbound_store.set_source_operation_coordinator(None)
    visible_versioned = [
        node
        for node in unbound_store.find_nodes(
            label_contains=f"{prefix}:versioned", limit=10
        )
        if node.id == f"{prefix}:versioned"
    ]
    assert visible_versioned == []
    legacy_metadata = {
        "tenant_id": harness.scope.tenant_id,
        "namespace": harness.scope.namespace,
        "workspace_id": harness.scope.workspace_id,
        "source_id": f"{prefix}-legacy",
        "chunk_ids": [f"{prefix}-legacy"],
    }
    unbound_store.upsert_node(
        GraphNode(f"{prefix}:legacy", f"{prefix}:legacy", metadata=legacy_metadata)
    )
    unbound_store.link_chunk(f"{prefix}:legacy", f"{prefix}-legacy")
    assert unbound_store.find_nodes(label_contains=f"{prefix}:legacy", limit=5)
    _purge_scope(unbound_store)
    return extra_scopes


def _run_contention_loop(harness: _Harness) -> None:
    graph = harness.graph
    source = f"{harness.source_prefix}-contention"
    current = _promote_generation(harness, source_id=source)
    _write_edge(
        harness,
        source_id=source,
        generation=current,
        from_id=f"{harness.source_prefix}:cont-a",
        to_id=f"{harness.source_prefix}:cont-b",
        chunk_id=f"{harness.source_prefix}-cont-0",
    )
    failures = 0
    for index in range(1, CONTENTION_ITERATIONS + 1):
        previous = current
        current = _promote_generation(harness, source_id=source)
        _write_edge(
            harness,
            source_id=source,
            generation=current,
            from_id=f"{harness.source_prefix}:cont-a",
            to_id=f"{harness.source_prefix}:cont-{index}",
            chunk_id=f"{harness.source_prefix}-cont-{index}",
        )
        _write_edge(
            harness,
            source_id=source,
            generation=previous,
            from_id=f"{harness.source_prefix}:stale-{index}",
            to_id=f"{harness.source_prefix}:cont-{index}",
            chunk_id=f"{harness.source_prefix}-stale-{index}",
        )
        visible = graph.find_nodes(
            label_contains=f"{harness.source_prefix}:cont-{index}",
            limit=5,
        )
        stale = graph.find_nodes(
            label_contains=f"{harness.source_prefix}:stale-{index}",
            limit=5,
        )
        if not visible or stale:
            failures += 1
    assert failures == 0, f"contention failures={failures}"


@pytest.fixture
def neo4j_live_enabled() -> None:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(
            f"set {RUN_ENV}=1 to run the Neo4j generation fencing qualification"
        )
    try:
        import neo4j

        assert neo4j.__version__ == "5.28.4"
    except ImportError as exc:
        pytest.skip(f"Neo4j driver unavailable: {exc}")


@pytest.mark.parametrize("run_index", [1, 2])
def test_neo4j_generation_fencing_live_qualification(
    neo4j_live_enabled: None,
    run_index: int,
) -> None:
    run_id = uuid.uuid4().hex
    try:
        harness = _open_neo4j_harness(run_id)
    except (IntegrationConfigurationError, ConnectionError, TimeoutError) as exc:
        pytest.skip(
            f"Neo4j backend unavailable during open: {type(exc).__name__}: {exc}"
        )

    extra_scopes: list[GraphScope] = []
    try:
        extra_scopes = _run_qualification_phases(harness)
        _run_contention_loop(harness)
        print(
            f"NEO4J_R2D2 run_id={run_id} run_index={run_index} "
            f"iterations={CONTENTION_ITERATIONS} failures=0 status=PASS"
        )
    finally:
        _cleanup_harness(harness, extra_scopes=extra_scopes)


def _inmemory_harness(run_id: str) -> _Harness:
    scope = GraphScope(
        f"r2d2-mem-{run_id}-tenant",
        namespace=f"r2d2-mem-{run_id}-namespace",
        workspace_id=f"r2d2-mem-{run_id}-workspace",
    )
    graph = InMemoryGraphStore()
    graph.bind_scope(scope)
    coordinator = InProcessSourceOperationCoordinator(owner_id=f"r2d2-mem-{run_id}")
    graph.set_source_operation_coordinator(coordinator)
    return _Harness(run_id, None, graph, coordinator, scope)


def test_generation_fencing_inmemory_parity() -> None:
    run_id = uuid.uuid4().hex
    harness = _inmemory_harness(run_id)
    _run_qualification_phases(harness)
    _run_contention_loop(harness)
