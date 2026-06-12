# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Sequence, Set

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore


class InMemoryGraphStore(GraphStore):
    def __init__(self, *, tenant_id: str | None = None) -> None:
        self._tenant_id = tenant_id.strip() if tenant_id else None
        self._nodes: Dict[str, GraphNode] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_by_node: Dict[str, Set[str]] = defaultdict(set)

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    def _tenant_matches(self, node: GraphNode) -> bool:
        if self._tenant_id is None:
            return True
        meta_tenant = str((node.metadata or {}).get("tenant_id", self._tenant_id))
        return meta_tenant == self._tenant_id

    def upsert_node(self, node: GraphNode) -> None:
        metadata = dict(node.metadata or {})
        if self._tenant_id is not None:
            metadata.setdefault("tenant_id", self._tenant_id)
        scoped = GraphNode(
            id=node.id,
            label=node.label,
            node_type=node.node_type,
            metadata=metadata,
        )
        self._nodes[node.id] = scoped

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._adj[edge.source_id].add(edge.target_id)
        self._adj[edge.target_id].add(edge.source_id)

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        self._chunk_by_node[node_id].add(chunk_id)

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        if node_id not in self._nodes:
            return []
        root = self._nodes[node_id]
        if not self._tenant_matches(root):
            return []
        seen = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        out: List[GraphNode] = []
        while queue:
            current, depth = queue.popleft()
            if depth > 0:
                node = self._nodes.get(current)
                if node is not None and self._tenant_matches(node):
                    out.append(node)
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
            if not self._tenant_matches(node):
                continue
            if needle in node.label.lower():
                found.append(node)
            if len(found) >= limit:
                break
        return found

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        ids: List[str] = []
        for nid in node_ids:
            if nid not in self._nodes:
                continue
            if not self._tenant_matches(self._nodes[nid]):
                continue
            ids.extend(sorted(self._chunk_by_node.get(nid, ())))
        return ids

    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        target = {cid.strip() for cid in chunk_ids if cid.strip()}
        if not target:
            return 0
        affected = 0
        for node_id, chunks in list(self._chunk_by_node.items()):
            before = len(chunks)
            chunks -= target
            affected += before - len(chunks)
            if not chunks:
                del self._chunk_by_node[node_id]
        orphan_nodes = [
            node_id
            for node_id in list(self._nodes.keys())
            if node_id not in self._chunk_by_node
        ]
        for node_id in orphan_nodes:
            del self._nodes[node_id]
            self._adj.pop(node_id, None)
            for neighbors in self._adj.values():
                neighbors.discard(node_id)
        return affected + len(orphan_nodes)

    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        scope = (tenant_id or self._tenant_id or "").strip()
        if not scope:
            count = len(self._nodes) + sum(len(v) for v in self._chunk_by_node.values())
            self._nodes.clear()
            self._adj.clear()
            self._chunk_by_node.clear()
            return count
        removed = 0
        for node_id in list(self._nodes.keys()):
            node = self._nodes[node_id]
            if str((node.metadata or {}).get("tenant_id", "")) != scope:
                continue
            del self._nodes[node_id]
            self._adj.pop(node_id, None)
            removed += 1
        for node_id in list(self._chunk_by_node.keys()):
            if node_id not in self._nodes:
                del self._chunk_by_node[node_id]
        return removed
