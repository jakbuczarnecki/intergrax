# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Heuristic entity/edge extraction for GraphRAG indexing (no fixed LLM provider)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from intergrax.distributed.source_operation import (
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore
from intergrax.rag.graph.indexer.validation import validate_graph_index_batch

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


class HeuristicGraphIndexer:
    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def index_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> int:
        validated_documents, resolved_ids = validate_graph_index_batch(
            self._store, documents, chunk_ids
        )
        count = 0
        for doc, chunk_id in zip(validated_documents, resolved_ids):
            count += self._index_one(doc, chunk_id)
        return count

    def _index_one(self, doc: KnowledgeDocument, chunk_id: str) -> int:
        text = doc.content
        entities = _ENTITY_RE.findall(text)[:12]
        if len(entities) < 2:
            return 0

        publication_metadata = self._publication_metadata(doc)
        node_ids: list[str] = []
        for label in entities:
            node_id = f"ent:{label.lower().replace(' ', '_')}"
            self._store.upsert_node(
                GraphNode(
                    id=node_id,
                    label=label,
                    node_type="entity",
                    metadata=publication_metadata,
                )
            )
            self._store.link_chunk(node_id, chunk_id)
            node_ids.append(node_id)

        for i in range(len(node_ids) - 1):
            self._store.upsert_edge(
                GraphEdge(
                    source_id=node_ids[i],
                    target_id=node_ids[i + 1],
                    relation="co_occurs",
                    metadata={"chunk_ids": [chunk_id], **publication_metadata},
                )
            )
        return len(node_ids)

    @staticmethod
    def _publication_metadata(doc: KnowledgeDocument) -> dict[str, object]:
        if SOURCE_PUBLICATION_GENERATION_METADATA_KEY not in doc.metadata:
            return {}
        return {
            SOURCE_PUBLICATION_GENERATION_METADATA_KEY: doc.metadata.get(
                SOURCE_PUBLICATION_GENERATION_METADATA_KEY
            ),
            "tenant_id": doc.scope.tenant_id,
            "namespace": doc.scope.namespace,
            "workspace_id": doc.scope.workspace_id,
            "source_id": str(doc.provenance.source_id),
        }
