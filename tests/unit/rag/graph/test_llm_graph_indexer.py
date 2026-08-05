# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.graph.indexer.llm_graph_indexer import LlmGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from tests.unit.rag.graph.fixtures import knowledge_document

pytestmark = pytest.mark.unit


class _FakeLlm:
    def generate_messages(self, messages, run_id: str = ""):
        del messages, run_id

        class _R:
            content = (
                '{"entities":[{"label":"Intergrax"},{"label":"Harness"}],'
                '"relations":[{"source":"Intergrax","target":"Harness","relation":"is_a"}]}'
            )

        return _R()


def test_llm_graph_indexer_extracts_entities() -> None:
    store = InMemoryGraphStore()
    indexer = LlmGraphIndexer(store, _FakeLlm())
    n = indexer.index_documents(
        [knowledge_document("Intergrax Harness provides agent runtime.")],
        chunk_ids=["c1"],
    )
    assert n >= 2
    assert store.chunk_ids_for_nodes({"ent:intergrax"})
