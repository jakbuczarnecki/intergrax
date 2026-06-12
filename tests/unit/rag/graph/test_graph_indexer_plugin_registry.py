# © Artur Czarnecki. All rights reserved.

import pytest
from langchain_core.documents import Document

from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.indexer.graph_indexer_factory import resolve_graph_indexer
from intergrax.rag.graph.indexer.plugin_registry import register_graph_indexer_plugin
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = pytest.mark.gate


def test_register_and_resolve_graph_indexer_plugin() -> None:
    def _factory(store, _profile, _llm):
        return HeuristicGraphIndexer(store)

    register_graph_indexer_plugin("lab_heuristic", _factory)
    profile = RagProfile(graph_indexer_plugin_id="lab_heuristic")
    indexer = resolve_graph_indexer(InMemoryGraphStore(), profile)
    count = indexer.index_documents(
        [Document(page_content="Acme Corp uses Intergrax Harness.")],
        chunk_ids=["chunk-plugin"],
    )
    assert count >= 1
