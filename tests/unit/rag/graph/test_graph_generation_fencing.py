# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.distributed.source_operation import (
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    SourceOperationCoordinator,
)
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphScope
from intergrax.rag.graph.generation_visibility import (
    graph_evidence_visible,
    resolve_scope_active_generations,
)
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore

pytestmark = pytest.mark.unit

TENANT = "gen-fence-tenant"
NAMESPACE = "gen-fence-ns"
WORKSPACE = "gen-fence-ws"
SOURCE_A = "source-a"
SOURCE_B = "source-b"
SCOPE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id=WORKSPACE)
OTHER_SCOPE = GraphScope(TENANT, namespace="other-ns", workspace_id=WORKSPACE)
OTHER_WORKSPACE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id="other-ws")
OTHER_TENANT = GraphScope("other-tenant", namespace=NAMESPACE, workspace_id=WORKSPACE)


def _source_key(source_id: str, scope: GraphScope = SCOPE) -> RagSourceOperationKey:
    return RagSourceOperationKey(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        workspace_id=scope.workspace_id,
        source_id=source_id,
    )


def _lease_generation(
    coordinator: InProcessSourceOperationCoordinator,
    source_id: str,
    *,
    scope: GraphScope = SCOPE,
    promote: bool = True,
) -> str:
    lease = coordinator.acquire(key=_source_key(source_id, scope))
    assert lease is not None
    generation = coordinator.publication_generation(lease=lease)
    if promote:
        assert coordinator.promote_publication(lease=lease)
    coordinator.release(lease=lease)
    return generation


def _metadata(
    source_id: str,
    generation: str,
    *,
    scope: GraphScope = SCOPE,
    chunk_id: str = "chunk-1",
) -> dict[str, object]:
    return {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
        "workspace_id": scope.workspace_id,
        "source_id": source_id,
        "chunk_ids": [chunk_id],
        SOURCE_PUBLICATION_GENERATION_METADATA_KEY: generation,
    }


def _write_source_edge(
    store: InMemoryGraphStore,
    *,
    source_id: str,
    generation: str,
    from_id: str,
    to_id: str,
    scope: GraphScope = SCOPE,
    chunk_id: str = "chunk-1",
) -> None:
    metadata = _metadata(source_id, generation, scope=scope, chunk_id=chunk_id)
    store.upsert_node(GraphNode(from_id, from_id, metadata=metadata))
    store.upsert_node(GraphNode(to_id, to_id, metadata=metadata))
    store.link_chunk(from_id, chunk_id)
    store.upsert_edge(GraphEdge(from_id, to_id, "supports", metadata=metadata))


def _bound_store(
    coordinator: SourceOperationCoordinator | None = None,
) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.bind_scope(SCOPE)
    if coordinator is not None:
        store.set_source_operation_coordinator(coordinator)
    return store


class _FailingCoordinator:
    def active_publication_generation(self, *, key: RagSourceOperationKey) -> str | None:
        del key
        raise RuntimeError("coordinator unavailable")


def test_g1_visible_when_g1_active() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=g1, from_id="ent:a", to_id="ent:b")

    assert store.find_nodes(label_contains="ent:a", limit=5)
    assert store.node_ids_for_chunks({"chunk-1"}) == {"ent:a"}
    assert any(node.id == "ent:b" for node in store.neighbors("ent:a"))


def test_promote_g2_hides_g1_immediately() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:g1",
        to_id="ent:shared",
    )

    g2 = _lease_generation(coordinator, SOURCE_A)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
        chunk_id="chunk-2",
    )

    assert store.find_nodes(label_contains="ent:g1", limit=5) == []
    assert store.node_ids_for_chunks({"chunk-1"}) == set()
    assert store.find_nodes(label_contains="ent:g2", limit=5)
    assert store.node_ids_for_chunks({"chunk-2"}) == {"ent:g2"}


def test_late_g1_write_after_g2_remains_invisible() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    g2 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:late-g1",
        to_id="ent:shared",
        chunk_id="chunk-late",
    )

    assert store.find_nodes(label_contains="ent:late-g1", limit=5) == []
    assert store.node_ids_for_chunks({"chunk-late"}) == set()
    assert store.find_nodes(label_contains="ent:g2", limit=5)


def test_g1_cleanup_preserves_g2() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    g2 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:g1",
        to_id="ent:shared",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
        chunk_id="chunk-2",
    )

    store.unlink_source_generation(SOURCE_A, g1, scope=SCOPE)

    assert store.find_nodes(label_contains="ent:g1", limit=5) == []
    assert store.find_nodes(label_contains="ent:g2", limit=5)
    assert store.node_ids_for_chunks({"chunk-2"}) == {"ent:g2"}


def test_shared_relation_stale_a_current_b_stays_visible() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    g2 = _lease_generation(coordinator, SOURCE_A)
    g_b = _lease_generation(coordinator, SOURCE_B)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:shared",
        to_id="ent:via-a",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:shared",
        to_id="ent:via-a2",
        chunk_id="chunk-a2",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_B,
        generation=g_b,
        from_id="ent:shared",
        to_id="ent:via-b",
        chunk_id="chunk-b",
    )

    assert any(node.id == "ent:via-b" for node in store.neighbors("ent:shared"))
    assert all(node.id != "ent:via-a" for node in store.neighbors("ent:shared"))


def test_remove_b_hides_relation_when_no_support_remains() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g2 = _lease_generation(coordinator, SOURCE_A)
    g_b = _lease_generation(coordinator, SOURCE_B)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:shared",
        to_id="ent:via-a2",
        chunk_id="chunk-a2",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_B,
        generation=g_b,
        from_id="ent:shared",
        to_id="ent:via-b",
        chunk_id="chunk-b",
    )

    store.unlink_source(SOURCE_B, scope=SCOPE)

    assert all(node.id != "ent:via-b" for node in store.neighbors("ent:shared"))
    assert any(node.id == "ent:via-a2" for node in store.neighbors("ent:shared"))


@pytest.mark.parametrize(
    ("scope", "chunk_id"),
    [
        (OTHER_SCOPE, "chunk-other-ns"),
        (OTHER_WORKSPACE, "chunk-other-ws"),
        (OTHER_TENANT, "chunk-other-tenant"),
    ],
)
def test_scope_isolation_is_unaffected(scope: GraphScope, chunk_id: str) -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g_main = _lease_generation(coordinator, SOURCE_A)
    g_other = _lease_generation(coordinator, SOURCE_A, scope=scope)
    main_store = _bound_store(coordinator)
    other_store = InMemoryGraphStore()
    other_store.bind_scope(scope)
    other_store.set_source_operation_coordinator(coordinator)

    _write_source_edge(
        main_store,
        source_id=SOURCE_A,
        generation=g_main,
        from_id="ent:main",
        to_id="ent:main-target",
    )
    _write_source_edge(
        other_store,
        source_id=SOURCE_A,
        generation=g_other,
        from_id="ent:other",
        to_id="ent:other-target",
        scope=scope,
        chunk_id=chunk_id,
    )

    assert main_store.find_nodes(label_contains="ent:main", limit=5)
    assert other_store.find_nodes(label_contains="ent:other", limit=5)
    assert main_store.find_nodes(label_contains="ent:other", limit=5) == []
    assert other_store.find_nodes(label_contains="ent:main", limit=5) == []


def test_repeated_g2_publication_is_idempotent() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g2 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
    )

    assert len(store.find_nodes(label_contains="ent:g2", limit=5)) == 1
    assert store.node_ids_for_chunks({"chunk-1"}) == {"ent:g2"}


def test_coordinator_failure_fails_closed() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(_FailingCoordinator())
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:a",
        to_id="ent:b",
    )

    assert store.find_nodes(label_contains="ent:a", limit=5) == []
    assert store.node_ids_for_chunks({"chunk-1"}) == set()


def test_generation_specific_unlink_removes_only_target_generation() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    g1 = _lease_generation(coordinator, SOURCE_A)
    g2 = _lease_generation(coordinator, SOURCE_A)
    store = _bound_store(coordinator)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g1,
        from_id="ent:g1",
        to_id="ent:shared",
    )
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        from_id="ent:g2",
        to_id="ent:shared",
        chunk_id="chunk-2",
    )

    removed = store.unlink_source_generation(SOURCE_A, g1, scope=SCOPE)

    assert removed > 0
    assert store.find_nodes(label_contains="ent:g1", limit=5) == []
    assert store.find_nodes(label_contains="ent:g2", limit=5)
    assert store.node_ids_for_chunks({"chunk-2"}) == {"ent:g2"}


def test_legacy_unversioned_evidence_stays_visible_without_coordinator() -> None:
    store = _bound_store(None)
    metadata = {
        "tenant_id": SCOPE.tenant_id,
        "namespace": SCOPE.namespace,
        "workspace_id": SCOPE.workspace_id,
        "source_id": SOURCE_A,
        "chunk_ids": ["legacy-chunk"],
    }
    store.upsert_node(GraphNode("ent:legacy", "ent:legacy", metadata=metadata))
    store.link_chunk("ent:legacy", "legacy-chunk")

    assert store.find_nodes(label_contains="ent:legacy", limit=5)
    assert store.node_ids_for_chunks({"legacy-chunk"}) == {"ent:legacy"}


def test_versioned_evidence_hidden_without_coordinator() -> None:
    store = _bound_store(None)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation="1:token",
        from_id="ent:versioned",
        to_id="ent:target",
    )

    assert store.find_nodes(label_contains="ent:versioned", limit=5) == []
    assert store.node_ids_for_chunks({"chunk-1"}) == set()


def test_resolve_scope_active_generations_raises_on_coordinator_failure() -> None:
    with pytest.raises(RuntimeError, match="graph generation resolution failed"):
        resolve_scope_active_generations(
            _FailingCoordinator(),
            SCOPE,
            [SOURCE_A],
        )


def test_graph_evidence_visible_matches_coordinator() -> None:
    coordinator = InProcessSourceOperationCoordinator(owner_id="test")
    generation = _lease_generation(coordinator, SOURCE_A)
    key = _source_key(SOURCE_A)
    assert graph_evidence_visible(
        versioned=True,
        generation=generation,
        source_key=key,
        coordinator=coordinator,
    )
    assert not graph_evidence_visible(
        versioned=True,
        generation="9:stale",
        source_key=key,
        coordinator=coordinator,
    )
