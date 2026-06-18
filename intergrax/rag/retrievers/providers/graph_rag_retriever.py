# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph-augmented retrieval: vector + keyword + graph channel fusion (M-RAG.42–43, M-RAG.53–54)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.graph.contracts.graph_store import GraphNode, GraphStore
from intergrax.rag.retrieval.graph_channel_fusion import (
    GraphChannelHit,
    build_keyword_hits,
    fuse_graph_channels,
)
from intergrax.rag.retrieval.graph_provenance_builder import build_graph_retrieval_provenance
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


@dataclass
class GraphRetrieverTrace:
    channel_contributions: Dict[str, List[str]] = field(default_factory=dict)
    expanded_node_ids: List[str] = field(default_factory=list)
    graph_provenance_summary: str = ""
    graph_provenance_records: List[Dict[str, Any]] = field(default_factory=list)


class GraphRagRetriever(BaseRetriever):
    STABILITY = "stable"

    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        graph_store: GraphStore,
        *,
        seed_top_k: int = 5,
        graph_hops: int = 1,
        hybrid_fusion_enabled: bool = True,
    ) -> None:
        self._vs = vector_store
        self._em = embedding_manager
        self._graph = graph_store
        self._seed_top_k = int(seed_top_k)
        self._graph_hops = int(graph_hops)
        self._hybrid_fusion_enabled = bool(hybrid_fusion_enabled)
        self._last_trace: GraphRetrieverTrace | None = None

    @property
    def last_graph_trace(self) -> GraphRetrieverTrace | None:
        return self._last_trace

    @classmethod
    def name(cls) -> str:
        return "graph_rag"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
        self._last_trace = None
        if not query.query_text:
            return []

        q_vec = (
            query.query_embedding
            if query.query_embedding is not None
            else self._em.embed_one(query.query_text)
        )
        seeds = self._vs.query(
            query_embedding=q_vec,
            top_k=self._seed_top_k,
            metadata_filter=query.metadata_filter,
            include_embeddings=False,
        )

        vector_hits: List[GraphChannelHit] = []
        by_id: Dict[str, RetrieverCandidate] = {}
        seed_chunk_ids: Set[str] = set()
        for hit in seeds:
            seed_chunk_ids.add(hit.id)
            vector_hits.append(
                GraphChannelHit(document_id=hit.id, score=float(hit.similarity_score), channel="vector")
            )
            by_id[hit.id] = RetrieverCandidate(
                id=hit.id,
                content=hit.content,
                metadata=dict(hit.metadata or {}),
                score=float(hit.similarity_score),
                rank=hit.rank,
            )

        related_node_ids: Set[str] = set()
        related_node_ids |= self._graph.node_ids_for_chunks(seed_chunk_ids)
        related_node_ids |= self._entity_ids_from_metadata(seeds)
        related_node_ids |= self._entity_ids_from_query_tokens(query.query_text)

        expanded_nodes: List[GraphNode] = []
        expanded_node_ids: Set[str] = set(related_node_ids)
        for node_id in list(related_node_ids):
            for neighbor in self._graph.neighbors(node_id, max_hops=self._graph_hops):
                expanded_node_ids.add(neighbor.id)
                expanded_nodes.append(neighbor)

        graph_hits: List[GraphChannelHit] = []
        extra_chunk_ids = self._graph.chunk_ids_for_nodes(expanded_node_ids)
        if extra_chunk_ids:
            prefetch = self._vs.query(
                query_embedding=q_vec,
                top_k=max(int(query.top_k), len(extra_chunk_ids) * 2),
                metadata_filter=query.metadata_filter,
            )
            for hit in prefetch:
                if hit.id not in extra_chunk_ids:
                    continue
                graph_hits.append(
                    GraphChannelHit(
                        document_id=hit.id,
                        score=float(hit.similarity_score) * 0.95,
                        channel="graph",
                    )
                )
                if hit.id not in by_id:
                    by_id[hit.id] = RetrieverCandidate(
                        id=hit.id,
                        content=hit.content,
                        metadata={**(hit.metadata or {}), "graph_expanded": True},
                        score=float(hit.similarity_score) * 0.95,
                        rank=hit.rank,
                    )

        keyword_hits = build_keyword_hits(
            query_text=query.query_text,
            candidates=[(candidate.id, candidate.content) for candidate in by_id.values()],
        )

        provenance_bundle = build_graph_retrieval_provenance(
            trace_id=query.query_text[:64] or "graph_rag",
            graph_id=attribute_access.optional(self._graph, "tenant_id", None) or "rag_graph",
            seed_node_ids=sorted(related_node_ids),
            expanded_nodes=expanded_nodes,
        )
        provenance_summary = provenance_bundle.explainability_summary
        provenance_records = [record.to_dict() for record in provenance_bundle.provenance_records]

        if self._hybrid_fusion_enabled and (vector_hits or graph_hits or keyword_hits):
            fusion = fuse_graph_channels(
                vector_hits=vector_hits,
                graph_hits=graph_hits,
                keyword_hits=keyword_hits,
                top_k=int(query.top_k),
            )
            ordered_ids = fusion.merged_document_ids
            candidates = [by_id[doc_id] for doc_id in ordered_ids if doc_id in by_id]
            self._last_trace = GraphRetrieverTrace(
                channel_contributions=dict(fusion.channel_contributions),
                expanded_node_ids=sorted(expanded_node_ids),
                graph_provenance_summary=provenance_summary,
                graph_provenance_records=provenance_records,
            )
            return candidates[: int(query.top_k)]

        candidates = list(by_id.values())
        candidates.sort(key=lambda c: c.score, reverse=True)
        self._last_trace = GraphRetrieverTrace(
            expanded_node_ids=sorted(expanded_node_ids),
            graph_provenance_summary=provenance_summary,
            graph_provenance_records=provenance_records,
        )
        return candidates[: int(query.top_k)]

    def _entity_ids_from_metadata(self, seeds) -> Set[str]:
        found: Set[str] = set()
        for hit in seeds:
            metadata = hit.metadata or {}
            raw_ids = metadata.get("graph_entity_ids")
            if isinstance(raw_ids, (list, tuple, set)):
                found |= {str(item) for item in raw_ids if str(item).strip()}
            raw_labels = metadata.get("graph_entities")
            if isinstance(raw_labels, (list, tuple, set)):
                for label in raw_labels:
                    label_text = str(label).strip()
                    if label_text:
                        found.add(f"ent:{label_text.lower().replace(' ', '_')}")
        return found

    def _entity_ids_from_query_tokens(self, query_text: str) -> Set[str]:
        tokens = [token.strip() for token in query_text.split() if len(token.strip()) >= 3]
        found: Set[str] = set()
        for token in tokens[:3]:
            for node in self._graph.find_nodes(label_contains=token, limit=3):
                found.add(node.id)
        return found
