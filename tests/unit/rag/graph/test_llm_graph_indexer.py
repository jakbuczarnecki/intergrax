# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.llm.messages import ChatMessage
from intergrax.rag.graph.indexer.llm_graph_indexer import LlmGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore

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
        [Document(page_content="Intergrax Harness provides agent runtime.")],
        chunk_ids=["c1"],
    )
    assert n >= 2
    assert store.chunk_ids_for_nodes({"ent:intergrax"})
