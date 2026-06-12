# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph store contract for GraphRAG (M-RAG.12) — backend-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class GraphNode:
    id: str
    label: str
    node_type: str = "entity"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str = "related_to"
    weight: float = 1.0
    metadata: Dict[str, object] = field(default_factory=dict)


class GraphStore(ABC):
    @property
    def tenant_id(self) -> str | None:
        """Optional tenant namespace for graph isolation (M-RAG.41)."""
        return None

    @abstractmethod
    def upsert_node(self, node: GraphNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        raise NotImplementedError

    @abstractmethod
    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        raise NotImplementedError

    @abstractmethod
    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        """Associate a graph node with a retrieval chunk/document id."""
        raise NotImplementedError

    @abstractmethod
    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        """Map graph nodes back to source chunk/document ids."""
        raise NotImplementedError

    @abstractmethod
    def node_ids_for_chunks(self, chunk_ids: Set[str]) -> Set[str]:
        """Map chunk/document ids to linked graph entity node ids."""
        raise NotImplementedError

    @abstractmethod
    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove HAS_CHUNK links for chunk ids and prune orphan entities (M-RAG.40)."""
        raise NotImplementedError

    @abstractmethod
    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        """Remove all RAG graph artifacts, optionally scoped to a tenant (M-RAG.40)."""
        raise NotImplementedError
