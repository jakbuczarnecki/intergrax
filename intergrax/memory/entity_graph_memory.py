# © Artur Czarnecki. All rights reserved.

"""Entity graph user memory — separate from document Graph RAG (Phase MEM-DEPTH-5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True, slots=True)
class EntityNode:
    entity_id: str
    label: str
    entity_type: str = "person"
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntityEdge:
    source_id: str
    target_id: str
    relation: str
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class EntityGraphMemoryStore:
    """
    In-process entity graph for user-scoped memory (not document Graph RAG).

    Production backends may replace this via ``intergrax.memory_stores`` EP.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, EntityNode] = {}
        self._edges: List[EntityEdge] = []

    def upsert_node(self, node: EntityNode) -> None:
        self._nodes[node.entity_id] = node

    def add_edge(self, edge: EntityEdge) -> None:
        self._edges.append(edge)

    def neighbors(self, entity_id: str) -> List[EntityNode]:
        related: Set[str] = set()
        for edge in self._edges:
            if edge.valid_until:
                continue
            if edge.source_id == entity_id:
                related.add(edge.target_id)
            if edge.target_id == entity_id:
                related.add(edge.source_id)
        return [self._nodes[node_id] for node_id in related if node_id in self._nodes]

    def list_nodes(self) -> List[EntityNode]:
        return list(self._nodes.values())
