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
        self._edge_keys: Set[tuple[str, str, str]] = set()
        self._unowned_edge_keys: Set[tuple[str, str, str]] = set()
        self._edge_chunks: Dict[tuple[str, str, str], Set[str]] = defaultdict(set)

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
        edge_key = (edge.source_id, edge.target_id, edge.relation)
        self._edge_keys.add(edge_key)
        raw_chunk_ids = (edge.metadata or {}).get("chunk_ids")
        if isinstance(raw_chunk_ids, (list, tuple, set)):
            self._edge_chunks[edge_key].update(
                str(chunk_id).strip()
                for chunk_id in raw_chunk_ids
                if str(chunk_id).strip()
            )
        else:
            self._unowned_edge_keys.add(edge_key)
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

    def node_ids_for_chunks(self, chunk_ids: Set[str]) -> Set[str]:
        targets = {cid.strip() for cid in chunk_ids if cid.strip()}
        if not targets:
            return set()
        found: Set[str] = set()
        for node_id, linked in self._chunk_by_node.items():
            if not linked & targets:
                continue
            node = self._nodes.get(node_id)
            if node is not None and self._tenant_matches(node):
                found.add(node_id)
        return found

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
        for edge_key, chunks in list(self._edge_chunks.items()):
            chunks -= target
            if chunks:
                continue
            del self._edge_chunks[edge_key]
            if edge_key not in self._unowned_edge_keys:
                self._edge_keys.discard(edge_key)
        orphan_nodes = [
            node_id
            for node_id in list(self._nodes.keys())
            if node_id not in self._chunk_by_node
        ]
        for node_id in orphan_nodes:
            del self._nodes[node_id]
            self._adj.pop(node_id, None)
        for edge_key in list(self._edge_keys):
            if edge_key[0] in orphan_nodes or edge_key[1] in orphan_nodes:
                self._edge_keys.discard(edge_key)
                self._unowned_edge_keys.discard(edge_key)
                self._edge_chunks.pop(edge_key, None)
        self._rebuild_adjacency()
        return affected + len(orphan_nodes)

    def _rebuild_adjacency(self) -> None:
        self._adj.clear()
        for source_id, target_id, _relation in self._edge_keys:
            if source_id not in self._nodes or target_id not in self._nodes:
                continue
            self._adj[source_id].add(target_id)
            self._adj[target_id].add(source_id)

    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        scope = (tenant_id or self._tenant_id or "").strip()
        if not scope:
            count = len(self._nodes) + sum(len(v) for v in self._chunk_by_node.values())
            self._nodes.clear()
            self._adj.clear()
            self._chunk_by_node.clear()
            self._edge_keys.clear()
            self._unowned_edge_keys.clear()
            self._edge_chunks.clear()
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
        for edge_key in list(self._edge_keys):
            if edge_key[0] not in self._nodes or edge_key[1] not in self._nodes:
                self._edge_keys.discard(edge_key)
                self._unowned_edge_keys.discard(edge_key)
                self._edge_chunks.pop(edge_key, None)
        self._rebuild_adjacency()
        return removed
