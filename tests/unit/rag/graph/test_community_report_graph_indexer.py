# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.indexer.community_report_graph_indexer import (
    CommunityReportGraphIndexer,
)
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from tests.unit.rag.graph.fixtures import knowledge_document

pytestmark = pytest.mark.gate


def test_community_report_indexer_creates_summary_node() -> None:
    store = InMemoryGraphStore()
    indexer = CommunityReportGraphIndexer(store)
    doc = knowledge_document(
        "Orion Analytics partners with Intergrax Harness for GraphRAG."
    )
    count = indexer.index_documents([doc], chunk_ids=["chunk-orion"])
    assert count >= 1
    nodes = store.find_nodes(label_contains="Orion", limit=5)
    assert nodes
    community = [node for node in store.find_nodes(label_contains="", limit=50) if node.node_type == "community_report"]
    assert community
