# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from intergrax.distributed.source_operation import (
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    SourceOperationCoordinator,
)
from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore


@dataclass(frozen=True, slots=True)
class _GraphEvidence:
    chunk_id: str
    generation: str | None
    source_key: RagSourceOperationKey | None
    versioned: bool


class InMemoryGraphStore(GraphStore):
    def __init__(self, *, tenant_id: str | None = None) -> None:
        self._tenant_id = tenant_id.strip() if tenant_id else None
        self._nodes: Dict[str, GraphNode] = {}
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_by_node: Dict[str, Set[str]] = defaultdict(set)
        self._node_evidence: Dict[str, Set[_GraphEvidence]] = defaultdict(set)
        self._versioned_node_ids: Set[str] = set()
        self._edge_keys: Set[tuple[str, str, str]] = set()
        self._unowned_edge_keys: Set[tuple[str, str, str]] = set()
        self._edge_evidence: Dict[
            tuple[str, str, str], Set[_GraphEvidence]
        ] = defaultdict(set)
        self._source_coordinator: SourceOperationCoordinator | None = None

    def set_source_operation_coordinator(
        self,
        coordinator: SourceOperationCoordinator | None,
    ) -> None:
        self._source_coordinator = coordinator

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
        if SOURCE_PUBLICATION_GENERATION_METADATA_KEY in metadata:
            self._versioned_node_ids.add(node.id)

    def upsert_edge(self, edge: GraphEdge) -> None:
        edge_key = (edge.source_id, edge.target_id, edge.relation)
        self._edge_keys.add(edge_key)
        raw_chunk_ids = (edge.metadata or {}).get("chunk_ids")
        if isinstance(raw_chunk_ids, (list, tuple, set)):
            for chunk_id in raw_chunk_ids:
                normalized_chunk_id = str(chunk_id).strip()
                if normalized_chunk_id:
                    self._edge_evidence[edge_key].add(
                        self._evidence(normalized_chunk_id, edge.metadata)
                    )
        else:
            if SOURCE_PUBLICATION_GENERATION_METADATA_KEY in (edge.metadata or {}):
                self._edge_evidence[edge_key].add(self._evidence("", edge.metadata))
            else:
                self._unowned_edge_keys.add(edge_key)
        self._adj[edge.source_id].add(edge.target_id)
        self._adj[edge.target_id].add(edge.source_id)

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id or node_id not in self._nodes:
            return
        self._chunk_by_node[node_id].add(normalized_chunk_id)
        self._node_evidence[node_id].add(
            self._evidence(normalized_chunk_id, self._nodes[node_id].metadata)
        )

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        if node_id not in self._nodes:
            return []
        root = self._nodes[node_id]
        if not self._node_visible(node_id) or not self._tenant_matches(root):
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
                if (
                    nxt in self._nodes
                    and self._node_visible(nxt)
                    and self._has_visible_edge(current, nxt)
                ):
                    queue.append((nxt, depth + 1))
        return out

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        needle = (label_contains or "").lower()
        found: List[GraphNode] = []
        for node_id, node in self._nodes.items():
            if not self._node_visible(node_id) or not self._tenant_matches(node):
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
            if not self._node_visible(nid) or not self._tenant_matches(self._nodes[nid]):
                continue
            ids.extend(
                sorted(
                    evidence.chunk_id
                    for evidence in self._node_evidence.get(nid, ())
                    if evidence.chunk_id and self._evidence_visible(evidence)
                )
            )
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
            if (
                node is not None
                and self._node_visible(node_id)
                and self._tenant_matches(node)
                and any(
                    evidence.chunk_id in targets
                    and self._evidence_visible(evidence)
                    for evidence in self._node_evidence.get(node_id, ())
                )
            ):
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
            self._node_evidence[node_id] = {
                evidence
                for evidence in self._node_evidence.get(node_id, ())
                if evidence.chunk_id not in target
            }
            if not self._node_evidence[node_id]:
                self._node_evidence.pop(node_id, None)
        for edge_key, evidence_set in list(self._edge_evidence.items()):
            remaining = {
                evidence
                for evidence in evidence_set
                if not evidence.chunk_id or evidence.chunk_id not in target
            }
            if remaining:
                self._edge_evidence[edge_key] = remaining
            else:
                del self._edge_evidence[edge_key]
                if edge_key not in self._unowned_edge_keys:
                    self._edge_keys.discard(edge_key)
        orphan_nodes = [
            node_id
            for node_id in list(self._nodes.keys())
            if node_id not in self._chunk_by_node
            and not self._node_evidence.get(node_id)
        ]
        for node_id in orphan_nodes:
            del self._nodes[node_id]
            self._adj.pop(node_id, None)
            self._node_evidence.pop(node_id, None)
            self._versioned_node_ids.discard(node_id)
        for edge_key in list(self._edge_keys):
            if edge_key[0] in orphan_nodes or edge_key[1] in orphan_nodes:
                self._edge_keys.discard(edge_key)
                self._unowned_edge_keys.discard(edge_key)
                self._edge_evidence.pop(edge_key, None)
        self._rebuild_adjacency()
        return affected + len(orphan_nodes)

    def _evidence(
        self,
        chunk_id: str,
        metadata: Dict[str, object] | None,
    ) -> _GraphEvidence:
        values = metadata or {}
        if SOURCE_PUBLICATION_GENERATION_METADATA_KEY not in values:
            return _GraphEvidence(chunk_id, None, None, False)
        generation = values.get(SOURCE_PUBLICATION_GENERATION_METADATA_KEY)
        source_key: RagSourceOperationKey | None = None
        try:
            if not isinstance(generation, str) or not generation.strip():
                raise ValueError
            tenant_id = values.get("tenant_id")
            source_id = values.get("source_id")
            namespace = values.get("namespace")
            workspace_id = values.get("workspace_id")
            if (
                not isinstance(tenant_id, str)
                or not isinstance(source_id, str)
                or (namespace is not None and not isinstance(namespace, str))
                or (workspace_id is not None and not isinstance(workspace_id, str))
            ):
                raise ValueError
            source_key = RagSourceOperationKey(
                tenant_id=tenant_id,
                namespace=namespace,
                workspace_id=workspace_id,
                source_id=source_id,
            )
        except (TypeError, ValueError):
            generation = None
        return _GraphEvidence(
            chunk_id,
            generation if isinstance(generation, str) else None,
            source_key,
            True,
        )

    def _evidence_visible(self, evidence: _GraphEvidence) -> bool:
        if not evidence.versioned:
            return True
        if (
            evidence.generation is None
            or evidence.source_key is None
            or self._source_coordinator is None
        ):
            return False
        try:
            active_generation = self._source_coordinator.active_publication_generation(
                key=evidence.source_key
            )
        except Exception:
            return False
        return active_generation == evidence.generation

    def _node_visible(self, node_id: str) -> bool:
        if node_id not in self._versioned_node_ids:
            return True
        return any(
            self._evidence_visible(evidence)
            for evidence in self._node_evidence.get(node_id, ())
        )

    def _has_visible_edge(self, source_id: str, target_id: str) -> bool:
        return any(
            edge_key[0] == source_id
            and edge_key[1] == target_id
            and self._edge_visible(edge_key)
            for edge_key in self._edge_keys
        ) or any(
            edge_key[0] == target_id
            and edge_key[1] == source_id
            and self._edge_visible(edge_key)
            for edge_key in self._edge_keys
        )

    def _edge_visible(self, edge_key: tuple[str, str, str]) -> bool:
        if edge_key in self._unowned_edge_keys:
            return True
        return any(
            self._evidence_visible(evidence)
            for evidence in self._edge_evidence.get(edge_key, ())
        )

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
            self._node_evidence.clear()
            self._versioned_node_ids.clear()
            self._edge_keys.clear()
            self._unowned_edge_keys.clear()
            self._edge_evidence.clear()
            return count
        removed = 0
        for node_id in list(self._nodes.keys()):
            node = self._nodes[node_id]
            if str((node.metadata or {}).get("tenant_id", "")) != scope:
                continue
            del self._nodes[node_id]
            self._adj.pop(node_id, None)
            self._versioned_node_ids.discard(node_id)
            removed += 1
        for node_id in list(self._chunk_by_node.keys()):
            if node_id not in self._nodes:
                del self._chunk_by_node[node_id]
        for node_id in list(self._node_evidence.keys()):
            if node_id not in self._nodes:
                del self._node_evidence[node_id]
        for edge_key in list(self._edge_keys):
            if edge_key[0] not in self._nodes or edge_key[1] not in self._nodes:
                self._edge_keys.discard(edge_key)
                self._unowned_edge_keys.discard(edge_key)
                self._edge_evidence.pop(edge_key, None)
        self._rebuild_adjacency()
        return removed
