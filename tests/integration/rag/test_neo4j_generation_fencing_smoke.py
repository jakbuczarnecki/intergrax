from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import pytest

from intergrax.distributed.source_operation import (
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.neo4j.bundle import (
    create_neo4j_graph_store,
)
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphScope
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.integration]

RUN_ENV = "INTERGRAX_RUN_NEO4J_LIVE"
TENANT = "neo4j-gen-fence"
NAMESPACE = "neo4j-gen-fence-ns"
WORKSPACE = "neo4j-gen-fence-ws"
SOURCE = "neo4j-gen-fence-source"
SCOPE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id=WORKSPACE)


@pytest.fixture
def neo4j_generation_store() -> Generator[Neo4jRagGraphStore, None, None]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the Neo4j generation fencing smoke")

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

    profile = RagProfile(graph_store_backend="neo4j", graph_rag_enabled=True)
    graph = create_rag_graph_store(
        profile=profile,
        integration_graph_store=integration,
        tenant_id=TENANT,
    )
    assert isinstance(graph, Neo4jRagGraphStore)
    graph.bind_scope(SCOPE)
    coordinator = InProcessSourceOperationCoordinator(
        owner_id=f"neo4j-gen-fence-{run_id}",
        token_factory=lambda: f"token-{run_id}",
    )
    graph.set_source_operation_coordinator(coordinator)

    try:
        yield graph
    finally:
        graph.purge_graph(tenant_id=TENANT)
        integration.close()


def _metadata(generation: str, *, chunk_id: str = "chunk-1") -> dict[str, object]:
    return {
        "tenant_id": SCOPE.tenant_id,
        "namespace": SCOPE.namespace,
        "workspace_id": SCOPE.workspace_id,
        "source_id": SOURCE,
        "chunk_ids": [chunk_id],
        SOURCE_PUBLICATION_GENERATION_METADATA_KEY: generation,
    }


def _promote_generation(
    coordinator: InProcessSourceOperationCoordinator,
    *,
    source_id: str = SOURCE,
) -> str:
    key = RagSourceOperationKey(
        tenant_id=SCOPE.tenant_id,
        namespace=SCOPE.namespace,
        workspace_id=SCOPE.workspace_id,
        source_id=source_id,
    )
    lease = coordinator.acquire(key=key)
    assert lease is not None
    generation = coordinator.publication_generation(lease=lease)
    assert coordinator.promote_publication(lease=lease)
    coordinator.release(lease=lease)
    return generation


def _write_topology(
    graph: Neo4jRagGraphStore,
    *,
    generation: str,
    from_id: str,
    to_id: str,
    chunk_id: str,
    metadata: dict[str, object] | None = None,
) -> None:
    resolved = dict(metadata or _metadata(generation, chunk_id=chunk_id))
    graph.upsert_node(GraphNode(from_id, from_id, metadata=resolved))
    graph.upsert_node(GraphNode(to_id, to_id, metadata=resolved))
    graph.link_chunk(from_id, chunk_id)
    graph.upsert_edge(GraphEdge(from_id, to_id, "supports", metadata=resolved))


def test_neo4j_generation_fencing_smoke(neo4j_generation_store: Neo4jRagGraphStore) -> None:
    graph = neo4j_generation_store
    coordinator = graph._source_coordinator  # noqa: SLF001 - smoke inspection
    assert coordinator is not None

    g1 = _promote_generation(coordinator)
    _write_topology(
        graph,
        generation=g1,
        from_id="ent:g1",
        to_id="ent:shared",
        chunk_id="chunk-g1",
    )
    assert graph.find_nodes(label_contains="ent:g1", limit=5)
    assert graph.node_ids_for_chunks({"chunk-g1"}) == {"ent:g1"}

    g2 = _promote_generation(coordinator)
    _write_topology(
        graph,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
        chunk_id="chunk-g2",
    )
    assert graph.find_nodes(label_contains="ent:g1", limit=5) == []
    assert graph.find_nodes(label_contains="ent:g2", limit=5)
    assert graph.node_ids_for_chunks({"chunk-g1"}) == set()
    assert graph.node_ids_for_chunks({"chunk-g2"}) == {"ent:g2"}

    _write_topology(
        graph,
        generation=g1,
        from_id="ent:late-g1",
        to_id="ent:shared",
        chunk_id="chunk-late-g1",
    )
    assert graph.find_nodes(label_contains="ent:late-g1", limit=5) == []
    assert graph.node_ids_for_chunks({"chunk-late-g1"}) == set()
    assert graph.find_nodes(label_contains="ent:g2", limit=5)

    graph.unlink_source_generation(SOURCE, g1, scope=SCOPE)
    assert graph.find_nodes(label_contains="ent:g2", limit=5)
    assert graph.node_ids_for_chunks({"chunk-g2"}) == {"ent:g2"}

    other_source = "neo4j-gen-fence-source-b"
    g_b = _promote_generation(coordinator, source_id=other_source)
    shared_metadata = _metadata(g_b, chunk_id="chunk-b")
    shared_metadata["source_id"] = other_source
    _write_topology(
        graph,
        generation=g_b,
        from_id="ent:shared",
        to_id="ent:via-b",
        chunk_id="chunk-b",
        metadata=shared_metadata,
    )
    assert any(node.id == "ent:via-b" for node in graph.neighbors("ent:shared"))
