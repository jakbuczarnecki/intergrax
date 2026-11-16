# © Artur Czarnecki. All rights reserved.

"""GraphRAG evidence resolves its active publication under the key scoped by the root document id (G-23).

Ingest promotes the publication of a source under a ``RagSourceOperationKey`` that carries ``publication_scope_id``
(the root document id). The graph side has to resolve it under the same key; a wrong or missing scope id, another
source, tenant, namespace or workspace must never become visible.
"""

from __future__ import annotations

import pytest
from intergrax.distributed.source_operation import (
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
)
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphScope
from intergrax.rag.graph.generation_visibility import (
    resolve_scope_active_generations,
    visibility_query_params,
)
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from tests.unit.rag.graph.fixtures import knowledge_document

pytestmark = pytest.mark.unit

TENANT = "scope-vis-tenant"
NAMESPACE = "scope-vis-ns"
WORKSPACE = "scope-vis-ws"
SOURCE_A = "source-a"
SOURCE_B = "source-b"
ROOT_A = "root-a"
SCOPE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id=WORKSPACE)
OTHER_SCOPE = GraphScope(TENANT, namespace="other-ns", workspace_id=WORKSPACE)
OTHER_WORKSPACE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id="other-ws")
OTHER_TENANT = GraphScope("other-tenant", namespace=NAMESPACE, workspace_id=WORKSPACE)
_ROOT_KEY = "root_document_id"


def _source_key(
    source_id: str,
    *,
    root: str | None,
    scope: GraphScope = SCOPE,
) -> RagSourceOperationKey:
    return RagSourceOperationKey(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        workspace_id=scope.workspace_id,
        source_id=source_id,
        publication_scope_id=root,
    )


def _promote(
    coordinator: InProcessSourceOperationCoordinator,
    source_id: str,
    *,
    root: str | None,
    scope: GraphScope = SCOPE,
) -> str:
    lease = coordinator.acquire(key=_source_key(source_id, root=root, scope=scope))
    assert lease is not None
    generation = coordinator.publication_generation(lease=lease)
    assert coordinator.promote_publication(lease=lease)
    coordinator.release(lease=lease)
    return generation


def _metadata(
    source_id: str,
    generation: str,
    *,
    root: str | None,
    scope: GraphScope = SCOPE,
    chunk_id: str = "chunk-1",
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
        "workspace_id": scope.workspace_id,
        "source_id": source_id,
        "chunk_ids": [chunk_id],
        SOURCE_PUBLICATION_GENERATION_METADATA_KEY: generation,
    }
    if root is not None:
        metadata[_ROOT_KEY] = root
    return metadata


def _write_source_edge(
    store: InMemoryGraphStore,
    *,
    source_id: str,
    generation: str,
    root: str | None,
    from_id: str,
    to_id: str,
    scope: GraphScope = SCOPE,
    chunk_id: str = "chunk-1",
) -> None:
    metadata = _metadata(source_id, generation, root=root, scope=scope, chunk_id=chunk_id)
    store.upsert_node(GraphNode(from_id, from_id, metadata=metadata))
    store.upsert_node(GraphNode(to_id, to_id, metadata=metadata))
    store.link_chunk(from_id, chunk_id)
    store.upsert_edge(GraphEdge(from_id, to_id, "supports", metadata=metadata))


def _bound_store(
    coordinator: InProcessSourceOperationCoordinator,
    scope: GraphScope = SCOPE,
) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.bind_scope(scope)
    store.set_source_operation_coordinator(coordinator)
    return store


def _coordinator() -> InProcessSourceOperationCoordinator:
    return InProcessSourceOperationCoordinator(owner_id="test")


def _is_visible(store: InMemoryGraphStore, label: str, chunk_id: str = "chunk-1") -> bool:
    found = bool(store.find_nodes(label_contains=label, limit=5))
    by_chunk = label in store.node_ids_for_chunks({chunk_id})
    assert found == by_chunk, (label, found, by_chunk)
    return found


# --- positives --------------------------------------------------------------------------------------------------


def test_scoped_publication_makes_stamped_evidence_visible() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=generation, root=ROOT_A, from_id="ent:a", to_id="ent:b")

    assert store.find_nodes(label_contains="ent:a", limit=5)
    assert store.node_ids_for_chunks({"chunk-1"}) == {"ent:a"}
    assert any(node.id == "ent:b" for node in store.neighbors("ent:a"))


def test_reingest_under_the_same_scoped_key_hides_the_first_generation() -> None:
    coordinator = _coordinator()
    g1 = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=g1, root=ROOT_A, from_id="ent:g1", to_id="ent:shared")
    assert _is_visible(store, "ent:g1")

    g2 = _promote(coordinator, SOURCE_A, root=ROOT_A)
    _write_source_edge(
        store,
        source_id=SOURCE_A,
        generation=g2,
        root=ROOT_A,
        from_id="ent:g2",
        to_id="ent:shared",
        chunk_id="chunk-2",
    )

    assert not _is_visible(store, "ent:g1")
    assert _is_visible(store, "ent:g2", "chunk-2")


# --- negatives: visibility is never widened ---------------------------------------------------------------------


def test_wrong_scope_id_stays_invisible() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=generation, root="root-x", from_id="ent:a", to_id="ent:b")

    assert not _is_visible(store, "ent:a")
    assert store.neighbors("ent:a") == []


def test_missing_scope_id_fails_closed_when_the_publication_is_scoped() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=generation, root=None, from_id="ent:a", to_id="ent:b")

    assert not _is_visible(store, "ent:a")


@pytest.mark.parametrize("blank", ["", "   ", 7])
def test_blank_or_non_string_scope_id_is_never_turned_into_a_key(blank: object) -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    metadata = _metadata(SOURCE_A, generation, root=None)
    metadata[_ROOT_KEY] = blank
    store.upsert_node(GraphNode("ent:a", "ent:a", metadata=metadata))
    store.link_chunk("ent:a", "chunk-1")

    assert not _is_visible(store, "ent:a")


def test_other_source_with_a_borrowed_scope_id_stays_invisible() -> None:
    coordinator = _coordinator()
    g_a = _promote(coordinator, SOURCE_A, root=ROOT_A)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=g_a, root=ROOT_A, from_id="ent:a", to_id="ent:a-target")
    # SOURCE_B has no publication; its evidence claims SOURCE_A's root and generation.
    _write_source_edge(
        store,
        source_id=SOURCE_B,
        generation=g_a,
        root=ROOT_A,
        from_id="ent:b",
        to_id="ent:b-target",
        chunk_id="chunk-b",
    )

    assert _is_visible(store, "ent:a")
    assert not _is_visible(store, "ent:b", "chunk-b")


@pytest.mark.parametrize(
    ("other", "chunk_id"),
    [
        (OTHER_TENANT, "chunk-other-tenant"),
        (OTHER_SCOPE, "chunk-other-ns"),
        (OTHER_WORKSPACE, "chunk-other-ws"),
    ],
)
def test_other_tenant_namespace_or_workspace_sees_nothing(other: GraphScope, chunk_id: str) -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)  # promoted under SCOPE only
    main_store = _bound_store(coordinator)
    other_store = _bound_store(coordinator, other)
    _write_source_edge(
        main_store, source_id=SOURCE_A, generation=generation, root=ROOT_A, from_id="ent:main", to_id="ent:m2"
    )
    _write_source_edge(
        other_store,
        source_id=SOURCE_A,
        generation=generation,
        root=ROOT_A,
        from_id="ent:other",
        to_id="ent:o2",
        scope=other,
        chunk_id=chunk_id,
    )

    assert _is_visible(main_store, "ent:main")
    assert not _is_visible(other_store, "ent:other", chunk_id)
    assert main_store.find_nodes(label_contains="ent:other", limit=5) == []
    assert other_store.find_nodes(label_contains="ent:main", limit=5) == []


# --- compatibility ----------------------------------------------------------------------------------------------


def test_unscoped_publication_and_evidence_without_scope_id_stay_visible() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=None)
    store = _bound_store(coordinator)
    _write_source_edge(store, source_id=SOURCE_A, generation=generation, root=None, from_id="ent:a", to_id="ent:b")

    assert _is_visible(store, "ent:a")


# --- shared helper ----------------------------------------------------------------------------------------------


def test_resolve_scope_active_generations_uses_the_scope_id_mapping() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)

    assert resolve_scope_active_generations(
        coordinator, SCOPE, [SOURCE_A], publication_scope_ids={SOURCE_A: ROOT_A}
    ) == {SOURCE_A: generation}
    assert resolve_scope_active_generations(
        coordinator, SCOPE, [f" {SOURCE_A} "], publication_scope_ids={SOURCE_A: ROOT_A}
    ) == {SOURCE_A: generation}


@pytest.mark.parametrize("mapping", [{SOURCE_A: "root-x"}, {SOURCE_A: "  "}, {SOURCE_B: ROOT_A}, {}, None])
def test_resolve_scope_active_generations_wrong_blank_or_no_scope_id_finds_nothing(
    mapping: dict[str, str] | None,
) -> None:
    coordinator = _coordinator()
    _promote(coordinator, SOURCE_A, root=ROOT_A)

    assert resolve_scope_active_generations(coordinator, SCOPE, [SOURCE_A], publication_scope_ids=mapping) == {}


def test_resolve_scope_active_generations_still_resolves_an_unscoped_publication() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=None)

    assert resolve_scope_active_generations(coordinator, SCOPE, [SOURCE_A]) == {SOURCE_A: generation}
    assert resolve_scope_active_generations(coordinator, SCOPE, [SOURCE_A], publication_scope_ids={SOURCE_A: "  "}) == {
        SOURCE_A: generation
    }


def test_visibility_query_params_passes_the_mapping_through() -> None:
    coordinator = _coordinator()
    generation = _promote(coordinator, SOURCE_A, root=ROOT_A)

    scoped = visibility_query_params(
        scope=SCOPE,
        coordinator=coordinator,
        source_ids=[SOURCE_A],
        publication_scope_ids={SOURCE_A: ROOT_A},
    )
    unscoped = visibility_query_params(scope=SCOPE, coordinator=coordinator, source_ids=[SOURCE_A])

    assert scoped["active_pairs"] == [{"source_id": SOURCE_A, "generation": generation}]
    assert unscoped["active_pairs"] == []


# --- carrier: the heuristic indexer stamps the root document id ------------------------------------------------


class _RecordingStore(InMemoryGraphStore):
    def __init__(self) -> None:
        super().__init__()
        self.node_metadata: list[dict[str, object]] = []
        self.edge_metadata: list[dict[str, object]] = []

    def upsert_node(self, node: GraphNode) -> None:
        self.node_metadata.append(dict(node.metadata))
        super().upsert_node(node)

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.edge_metadata.append(dict(edge.metadata))
        super().upsert_edge(edge)


_CONTENT = "Alice Smith met Bob Jones near Grand Central."
_DOC_SCOPE = GraphScope(TENANT, namespace=NAMESPACE, workspace_id=WORKSPACE)


def _document(root: str, generation: str | None = None):
    metadata: dict[str, object] = {}
    if generation is not None:
        metadata[SOURCE_PUBLICATION_GENERATION_METADATA_KEY] = generation
    return knowledge_document(
        _CONTENT,
        tenant_id=TENANT,
        namespace=NAMESPACE,
        workspace_id=WORKSPACE,
        document_id=root,
        metadata=metadata,
    )


def test_indexer_stamps_the_root_document_id_on_versioned_evidence() -> None:
    store = _RecordingStore()
    HeuristicGraphIndexer(store).index_documents([_document("root-doc", "1:token")], chunk_ids=["c-1"])

    assert store.node_metadata and store.edge_metadata
    for metadata in (*store.node_metadata, *store.edge_metadata):
        assert metadata[_ROOT_KEY] == "root-doc"
        assert metadata[SOURCE_PUBLICATION_GENERATION_METADATA_KEY] == "1:token"


def test_indexer_writes_no_root_document_id_on_unversioned_evidence() -> None:
    store = _RecordingStore()
    HeuristicGraphIndexer(store).index_documents([_document("root-doc")], chunk_ids=["c-1"])

    assert store.node_metadata and store.edge_metadata
    for metadata in (*store.node_metadata, *store.edge_metadata):
        assert _ROOT_KEY not in metadata
        assert SOURCE_PUBLICATION_GENERATION_METADATA_KEY not in metadata


def test_indexer_evidence_is_visible_under_its_root_and_invisible_under_another() -> None:
    root = "root-doc"
    source_id = f"source:{root}"
    coordinator = _coordinator()
    generation = _promote(coordinator, source_id, root=root, scope=_DOC_SCOPE)
    store = _bound_store(coordinator, _DOC_SCOPE)
    HeuristicGraphIndexer(store).index_documents([_document(root, generation)], chunk_ids=["c-1"])
    assert store.find_nodes(label_contains="Alice Smith", limit=5)
    assert store.node_ids_for_chunks({"c-1"})

    other_coordinator = _coordinator()
    other_generation = _promote(other_coordinator, source_id, root="another-root", scope=_DOC_SCOPE)
    other_store = _bound_store(other_coordinator, _DOC_SCOPE)
    HeuristicGraphIndexer(other_store).index_documents([_document(root, other_generation)], chunk_ids=["c-1"])
    assert other_store.find_nodes(label_contains="Alice Smith", limit=5) == []
    assert other_store.node_ids_for_chunks({"c-1"}) == set()
