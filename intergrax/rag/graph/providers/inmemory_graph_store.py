# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_by_node: Dict[str, Set[str]] = defaultdict(set)

    def upsert_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._adj[edge.source_id].add(edge.target_id)
        self._adj[edge.target_id].add(edge.source_id)

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        self._chunk_by_node[node_id].add(chunk_id)

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        if node_id not in self._nodes:
            return []
        seen = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        out: List[GraphNode] = []
        while queue:
            current, depth = queue.popleft()
            if depth > 0:
                out.append(self._nodes[current])
            if depth >= max_hops:
                continue
            for nxt in self._adj.get(current, ()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in self._nodes:
                    queue.append((nxt, depth + 1))
        return out

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        needle = (label_contains or "").lower()
        found: List[GraphNode] = []
        for node in self._nodes.values():
            if needle in node.label.lower():
                found.append(node)
            if len(found) >= limit:
                break
        return found

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        ids: List[str] = []
        for nid in node_ids:
            ids.extend(sorted(self._chunk_by_node.get(nid, ())))
        return ids
