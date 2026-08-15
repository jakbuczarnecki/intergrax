# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness-native community-report graph indexer mode (M-RAG.47) — opt-in, no MS GraphRAG vendoring."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.indexer.validation import validate_graph_index_batch


class CommunityReportGraphIndexer:
    """
    Build entity graph via heuristic indexer, then store a short community summary node.

    When ``LLMAdapter`` is provided, summary text is LLM-generated; otherwise deterministic excerpt.
    """

    def __init__(self, store: GraphStore, llm: LLMAdapter | None = None) -> None:
        self._store = store
        self._llm = llm
        self._heuristic = HeuristicGraphIndexer(store)

    def index_documents(
        self,
        documents: Sequence[KnowledgeDocument],
        *,
        chunk_ids: Sequence[str] | None = None,
    ) -> int:
        validated_documents, resolved_ids = validate_graph_index_batch(
            self._store, documents, chunk_ids
        )
        count = self._heuristic.index_documents(
            validated_documents, chunk_ids=resolved_ids
        )
        for doc, chunk_id in zip(validated_documents, resolved_ids):
            count += self._index_community_report(doc, chunk_id)
        return count

    def _index_community_report(
        self, doc: KnowledgeDocument, chunk_id: str
    ) -> int:
        text = doc.content.strip()
        if not text:
            return 0
        summary = self._summarize(text)
        community_id = f"community:{chunk_id}"
        self._store.upsert_node(
            GraphNode(
                id=community_id,
                label=summary[:120],
                node_type="community_report",
                metadata={"summary": summary, "source_chunk_id": chunk_id},
            )
        )
        self._store.link_chunk(community_id, chunk_id)
        self._store.upsert_edge(
            GraphEdge(source_id=community_id, target_id=community_id, relation="community_summary")
        )
        return 1

    def _summarize(self, text: str) -> str:
        if self._llm is None:
            return text[:280]
        prompt = (
            "Summarize the following document excerpt as a one-sentence community report "
            "for a knowledge graph index:\n\n"
            f"{text[:4000]}"
        )
        response = self._llm.generate_messages(
            [ChatMessage(role="user", content=prompt)],
            run_id="rag-community-report-index",
        )
        body = (response.content or "").strip()
        return body or text[:280]
