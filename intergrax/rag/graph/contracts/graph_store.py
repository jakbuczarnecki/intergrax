# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph store contract for GraphRAG (M-RAG.12) — backend-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


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
    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        """Map graph nodes back to source chunk/document ids."""
        raise NotImplementedError
