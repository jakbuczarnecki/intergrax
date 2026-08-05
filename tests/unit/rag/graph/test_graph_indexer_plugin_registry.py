# © Artur Czarnecki. All rights reserved.

from typing import get_type_hints

import pytest

from intergrax.rag.graph.indexer.graph_indexer_factory import (
    GraphIndexer,
    resolve_graph_indexer,
)
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.indexer.plugin_registry import (
    GraphIndexerPlugin,
    register_graph_indexer_plugin,
)
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from tests.unit.rag.graph.fixtures import knowledge_document

pytestmark = pytest.mark.gate


def test_register_and_resolve_graph_indexer_plugin() -> None:
    def _factory(store, _profile, _llm):
        return HeuristicGraphIndexer(store)

    register_graph_indexer_plugin("lab_heuristic", _factory)
    profile = RagProfile(graph_indexer_plugin_id="lab_heuristic")
    indexer = resolve_graph_indexer(InMemoryGraphStore(), profile)
    count = indexer.index_documents(
        [
            knowledge_document(
                "Acme Corp uses Intergrax Harness.",
                namespace="namespace-a",
                workspace_id="workspace-a",
            )
        ],
        chunk_ids=["chunk-plugin"],
    )
    assert count >= 1
    with pytest.raises(ValueError, match="cannot change"):
        indexer.index_documents(
            [
                knowledge_document(
                    "Acme Corp uses Intergrax Harness.",
                    namespace="namespace-b",
                    workspace_id="workspace-a",
                )
            ],
            chunk_ids=["chunk-plugin-2"],
        )


def test_graph_indexer_contracts_are_native() -> None:
    factory_hints = get_type_hints(GraphIndexer.index_documents)
    plugin_hints = get_type_hints(GraphIndexerPlugin.index_documents)

    assert factory_hints["documents"] == plugin_hints["documents"]
    assert "KnowledgeDocument" in str(factory_hints["documents"])
    assert "Document" not in str(factory_hints["documents"]).replace("KnowledgeDocument", "")


def test_graph_indexer_validates_batch_before_first_write() -> None:
    store = InMemoryGraphStore(tenant_id="tenant-a")
    valid = knowledge_document(
        "Acme Corp uses Intergrax Harness.",
        tenant_id="tenant-a",
        document_id="stable-doc",
    )
    indexer = HeuristicGraphIndexer(store)

    assert indexer.index_documents([]) == 0
    with pytest.raises(ValueError, match="chunk_ids length"):
        indexer.index_documents([valid], chunk_ids=[])
    with pytest.raises(ValueError, match="non-empty strings"):
        indexer.index_documents([valid], chunk_ids=[""])
    with pytest.raises(ValueError, match="tenant"):
        indexer.index_documents(
            [
                valid,
                knowledge_document(
                    "Beta Labs uses Intergrax Harness.",
                    tenant_id="tenant-b",
                    document_id="other-doc",
                ),
            ],
            chunk_ids=["chunk-a", "chunk-b"],
        )
    with pytest.raises(ValueError, match="unique"):
        indexer.index_documents(
            [
                valid,
                knowledge_document(
                    "Beta Labs uses Intergrax Harness.",
                    tenant_id="tenant-a",
                    document_id="other-doc",
                ),
            ],
            chunk_ids=["duplicate", "duplicate"],
        )
    with pytest.raises(ValueError, match="tenant and namespace"):
        indexer.index_documents(
            [
                knowledge_document(
                    "Acme Corp uses Intergrax Harness.",
                    tenant_id="tenant-a",
                    namespace="namespace-a",
                ),
                knowledge_document(
                    "Beta Labs uses Intergrax Harness.",
                    tenant_id="tenant-a",
                    namespace="namespace-b",
                    document_id="other-doc",
                ),
            ],
            chunk_ids=["chunk-a", "chunk-b"],
        )
    with pytest.raises(TypeError, match="KnowledgeDocument"):
        indexer.index_documents([valid, object()], chunk_ids=["chunk-a", "chunk-b"])  # type: ignore[list-item]

    assert store.find_nodes(label_contains="Acme", limit=5) == []


def test_graph_indexer_scope_fence_rejects_workspace_mixing_before_write() -> None:
    store = InMemoryGraphStore(tenant_id="tenant-a")
    indexer = HeuristicGraphIndexer(store)
    writes: list[str] = []
    original_upsert_node = store.upsert_node

    def record_upsert_node(node) -> None:
        writes.append(node.id)
        original_upsert_node(node)

    store.upsert_node = record_upsert_node  # type: ignore[method-assign]
    try:
        assert indexer.index_documents([]) == 0
        with pytest.raises(ValueError, match="chunk_ids length"):
            indexer.index_documents(
                [
                    knowledge_document(
                        "Acme Corp uses Intergrax Harness.",
                        namespace="namespace-a",
                        workspace_id="workspace-a",
                    )
                ],
                chunk_ids=[],
            )
        assert writes == []

        with pytest.raises(ValueError, match="workspace"):
            indexer.index_documents(
                [
                    knowledge_document(
                        "Acme Corp uses Intergrax Harness.",
                        workspace_id=None,
                    ),
                    knowledge_document(
                        "Beta Labs uses Intergrax Harness.",
                        workspace_id="workspace-a",
                        document_id="other-doc",
                    ),
                ],
                chunk_ids=["chunk-a", "chunk-b"],
            )
        assert writes == []

        valid = knowledge_document(
            "Acme Corp uses Intergrax Harness.",
            namespace="namespace-a",
            workspace_id="workspace-a",
        )
        assert indexer.index_documents([valid], chunk_ids=["chunk-valid"]) >= 1
        write_count = len(writes)
        assert indexer.index_documents([valid], chunk_ids=["chunk-valid-2"]) >= 1
        assert len(writes) > write_count
        writes_before_scope_conflicts = len(writes)

        with pytest.raises(ValueError, match="cannot change"):
            indexer.index_documents(
                [
                    knowledge_document(
                        "Beta Labs uses Intergrax Harness.",
                        namespace="namespace-b",
                        workspace_id="workspace-a",
                        document_id="namespace-doc",
                    )
                ],
                chunk_ids=["chunk-namespace"],
            )
        with pytest.raises(ValueError, match="cannot change"):
            indexer.index_documents(
                [
                    knowledge_document(
                        "Gamma Systems uses Intergrax Harness.",
                        namespace="namespace-a",
                        workspace_id="workspace-b",
                        document_id="workspace-doc",
                    )
                ],
                chunk_ids=["chunk-workspace"],
            )
        assert len(writes) == writes_before_scope_conflicts
    finally:
        store.upsert_node = original_upsert_node  # type: ignore[method-assign]


def test_graph_indexer_uses_stable_identity_without_mutating_document() -> None:
    store = InMemoryGraphStore(tenant_id="tenant-a")
    document = knowledge_document(
        "Acme Corp uses Intergrax Harness.",
        tenant_id="tenant-a",
        document_id="stable-doc",
    )
    before = document.model_dump(mode="python")

    assert HeuristicGraphIndexer(store).index_documents([document]) >= 1

    assert store.chunk_ids_for_nodes({"ent:acme_corp"}) == ["stable-doc"]
    assert document.identity.document_id == "stable-doc"
    assert document.scope.tenant_id == "tenant-a"
    assert document.provenance.source_id == "source:stable-doc"
    assert document.model_dump(mode="python") == before
