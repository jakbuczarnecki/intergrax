# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Heuristic entity/edge extraction for GraphRAG indexing (no fixed LLM provider)."""

from __future__ import annotations

import re
import uuid
from typing import Iterable, List, Sequence

from langchain_core.documents import Document

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


class HeuristicGraphIndexer:
    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def index_documents(self, documents: Sequence[Document], *, chunk_ids: Sequence[str] | None = None) -> int:
        count = 0
        ids = list(chunk_ids) if chunk_ids else [str(uuid.uuid4()) for _ in documents]
        for doc, chunk_id in zip(documents, ids):
            count += self._index_one(doc, chunk_id)
        return count

    def _index_one(self, doc: Document, chunk_id: str) -> int:
        text = doc.page_content or ""
        entities = _ENTITY_RE.findall(text)[:12]
        if len(entities) < 2:
            return 0

        node_ids: List[str] = []
        for label in entities:
            node_id = f"ent:{label.lower().replace(' ', '_')}"
            self._store.upsert_node(GraphNode(id=node_id, label=label, node_type="entity"))
            if isinstance(self._store, InMemoryGraphStore):
                self._store.link_chunk(node_id, chunk_id)
            node_ids.append(node_id)

        for i in range(len(node_ids) - 1):
            self._store.upsert_edge(
                GraphEdge(source_id=node_ids[i], target_id=node_ids[i + 1], relation="co_occurs")
            )
        return len(node_ids)
