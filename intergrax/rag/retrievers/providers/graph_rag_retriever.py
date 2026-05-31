# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph-augmented retrieval: vector seeds + 1-hop graph expansion."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverCandidate,
    RetrieverQuery,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager


class GraphRagRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: BaseVectorstoreManager,
        embedding_manager: BaseEmbeddingManager,
        graph_store: GraphStore,
        *,
        seed_top_k: int = 5,
        graph_hops: int = 1,
    ) -> None:
        self._vs = vector_store
        self._em = embedding_manager
        self._graph = graph_store
        self._seed_top_k = int(seed_top_k)
        self._graph_hops = int(graph_hops)

    @classmethod
    def name(cls) -> str:
        return "graph_rag"

    def retrieve(self, query: RetrieverQuery) -> List[RetrieverCandidate]:
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

        by_id: Dict[str, RetrieverCandidate] = {}
        for hit in seeds:
            by_id[hit.id] = RetrieverCandidate(
                id=hit.id,
                content=hit.content,
                metadata=dict(hit.metadata or {}),
                score=float(hit.similarity_score),
                rank=hit.rank,
            )

        seed_nodes = self._graph.find_nodes(label_contains=query.query_text.split()[0] if query.query_text else "", limit=5)
        expanded_ids: Set[str] = set(by_id.keys())
        related_node_ids: Set[str] = set()

        for hit in seeds:
            for node in self._graph.find_nodes(label_contains=(hit.content or "")[:40], limit=3):
                related_node_ids.add(node.id)
                for neighbor in self._graph.neighbors(node.id, max_hops=self._graph_hops):
                    related_node_ids.add(neighbor.id)

        for node in seed_nodes:
            related_node_ids.add(node.id)
            for neighbor in self._graph.neighbors(node.id, max_hops=self._graph_hops):
                related_node_ids.add(neighbor.id)

        extra_chunk_ids = self._graph.chunk_ids_for_nodes(related_node_ids)
        if extra_chunk_ids and hasattr(self._vs, "query"):
            prefetch = self._vs.query(
                query_embedding=q_vec,
                top_k=max(int(query.top_k), len(extra_chunk_ids) * 2),
                metadata_filter=query.metadata_filter,
            )
            for hit in prefetch:
                if hit.id in extra_chunk_ids and hit.id not in by_id:
                    by_id[hit.id] = RetrieverCandidate(
                        id=hit.id,
                        content=hit.content,
                        metadata={**(hit.metadata or {}), "graph_expanded": True},
                        score=float(hit.similarity_score) * 0.95,
                        rank=hit.rank,
                    )

        candidates = list(by_id.values())
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: int(query.top_k)]
