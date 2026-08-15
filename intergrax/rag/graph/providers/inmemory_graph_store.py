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
from intergrax.rag.graph.contracts.graph_store import (
    GraphEdge,
    GraphNode,
    GraphScope,
    GraphStore,
)
from intergrax.rag.graph.generation_visibility import graph_evidence_visible


@dataclass(frozen=True, slots=True)
class _GraphEvidence:
    scope_key: str
    source_id: str
    chunk_id: str
    generation: str | None
    source_key: RagSourceOperationKey | None
    versioned: bool


class _ScopedNodeMap(dict):
    """Keep private logical-id membership compatible with old harness probes."""

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        return isinstance(key, str) and any(
            isinstance(item, tuple) and len(item) == 2 and item[1] == key
            for item in self
        )


class InMemoryGraphStore(GraphStore):
    """Reference implementation of the scoped evidence ownership law."""

    def __init__(
        self,
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._tenant_id = tenant_id.strip() if tenant_id else None
        self._bound_scope: GraphScope | None = None
        if namespace is not None or workspace_id is not None:
            if self._tenant_id is None:
                raise ValueError("namespace/workspace require tenant_id")
            self._bound_scope = GraphScope(
                self._tenant_id, namespace=namespace, workspace_id=workspace_id
            )
        self._nodes: Dict[tuple[str, str], GraphNode] = _ScopedNodeMap()
        self._adj: Dict[tuple[str, str], Set[tuple[str, str]]] = defaultdict(set)
        self._node_evidence: Dict[
            tuple[str, str], Set[_GraphEvidence]
        ] = defaultdict(set)
        self._edge_keys: Set[tuple[str, str, str, str]] = set()
        self._edge_evidence: Dict[
            tuple[str, str, str, str], Set[_GraphEvidence]
        ] = defaultdict(set)
        self._source_coordinator: SourceOperationCoordinator | None = None

    def set_source_operation_coordinator(
        self, coordinator: SourceOperationCoordinator | None
    ) -> None:
        self._source_coordinator = coordinator

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    @property
    def scope(self) -> GraphScope | None:
        return self._bound_scope

    def bind_scope(self, scope: GraphScope) -> None:
        normalized = GraphScope.from_object(scope)
        if self._tenant_id is not None and normalized.tenant_id != self._tenant_id:
            raise ValueError("document tenant_id differs from bound graph store")
        if self._bound_scope is not None and self._bound_scope != normalized:
            raise ValueError("graph scope cannot change after binding")
        self._bound_scope = normalized

    def _scope_for_metadata(
        self,
        metadata: Dict[str, object] | None = None,
        scope: GraphScope | None = None,
    ) -> GraphScope:
        values = metadata or {}
        if self._bound_scope is not None:
            resolved = self._bound_scope
            for name in ("tenant_id", "namespace", "workspace_id"):
                supplied = values.get(name)
                if supplied is not None and supplied != object.__getattribute__(resolved, name):
                    raise ValueError(f"graph {name} differs from bound scope")
        elif scope is not None:
            resolved = GraphScope.from_object(scope)
        else:
            resolved = GraphScope(
                str(values.get("tenant_id") or self._tenant_id or "__legacy__"),
                namespace=values.get("namespace")
                if isinstance(values.get("namespace"), str)
                else None,
                workspace_id=values.get("workspace_id")
                if isinstance(values.get("workspace_id"), str)
                else None,
            )
        self.bind_scope(resolved)
        return resolved

    @staticmethod
    def _metadata(metadata: Dict[str, object] | None, scope: GraphScope) -> dict[str, object]:
        result = dict(metadata or {})
        result.update(
            {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            }
        )
        return result

    def _node_key(self, node_id: str, scope: GraphScope) -> tuple[str, str]:
        return (scope.key, node_id)

    def upsert_node(self, node: GraphNode) -> None:
        scope = self._scope_for_metadata(node.metadata)
        self._nodes[self._node_key(node.id, scope)] = GraphNode(
            id=node.id,
            label=node.label,
            node_type=node.node_type,
            metadata=self._metadata(node.metadata, scope),
        )

    def upsert_edge(self, edge: GraphEdge) -> None:
        metadata = dict(edge.metadata or {})
        scope = self._scope_for_metadata(metadata)
        edge_key = (scope.key, edge.source_id, edge.target_id, edge.relation)
        self._edge_keys.add(edge_key)
        raw_chunks = metadata.get("chunk_ids")
        chunks = (
            [str(item).strip() for item in raw_chunks if str(item).strip()]
            if isinstance(raw_chunks, (list, tuple, set))
            else [""]
        )
        source_id = str(metadata.get("source_id") or "__legacy__")
        self._edge_evidence[edge_key].update(
            self._evidence(chunk_id, metadata, scope, source_id)
            for chunk_id in chunks
        )
        source_key = self._node_key(edge.source_id, scope)
        target_key = self._node_key(edge.target_id, scope)
        self._adj[source_key].add(target_key)
        self._adj[target_key].add(source_key)

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        chunk_id = str(chunk_id).strip()
        if not chunk_id:
            return
        scope = self._scope_for_metadata()
        key = self._node_key(node_id, scope)
        node = self._nodes.get(key)
        if node is None:
            return
        self._node_evidence[key].add(
            self._evidence(chunk_id, node.metadata, scope, None)
        )

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        scope = self._scope_for_metadata()
        root = self._node_key(node_id, scope)
        if root not in self._nodes or not self._node_visible(root):
            return []
        hops = max(1, int(max_hops))
        seen = {root}
        queue: deque[tuple[tuple[str, str], int]] = deque([(root, 0)])
        out: List[GraphNode] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= hops:
                continue
            for nxt in self._adj.get(current, ()):
                if nxt in seen or not self._node_visible(nxt):
                    continue
                if not self._has_visible_edge(current, nxt):
                    continue
                seen.add(nxt)
                queue.append((nxt, depth + 1))
                out.append(self._nodes[nxt])
        return out

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        scope = self._scope_for_metadata()
        needle = (label_contains or "").lower()
        found: List[GraphNode] = []
        for key, node in self._nodes.items():
            if key[0] != scope.key or not self._node_visible(key):
                continue
            if needle in node.label.lower():
                found.append(node)
            if len(found) >= limit:
                break
        return found

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        scope = self._scope_for_metadata()
        ids: List[str] = []
        for node_id in node_ids:
            key = self._node_key(node_id, scope)
            if key not in self._nodes or not self._node_visible(key):
                continue
            ids.extend(
                sorted(
                    evidence.chunk_id
                    for evidence in self._node_evidence.get(key, ())
                    if evidence.chunk_id and self._evidence_visible(evidence)
                )
            )
        return ids

    def node_ids_for_chunks(self, chunk_ids: Set[str]) -> Set[str]:
        targets = {cid.strip() for cid in chunk_ids if cid.strip()}
        if not targets:
            return set()
        scope = self._scope_for_metadata()
        return {
            node_id
            for (scope_key, node_id), evidence_set in self._node_evidence.items()
            if scope_key == scope.key
            and any(
                evidence.chunk_id in targets and self._evidence_visible(evidence)
                for evidence in evidence_set
            )
        }

    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        target = {cid.strip() for cid in chunk_ids if cid.strip()}
        if not target:
            return 0
        scope = self._scope_for_metadata()
        removed = self._remove_evidence(
            scope.key,
            lambda evidence: evidence.chunk_id in target,
        )
        return removed + self._prune(scope.key)

    def unlink_source(self, source_id: str, *, scope: GraphScope | None = None) -> int:
        source_id = source_id.strip()
        if not source_id:
            return 0
        resolved = self._scope_for_metadata(scope=scope)
        removed = self._remove_evidence(
            resolved.key,
            lambda evidence: evidence.source_id == source_id,
        )
        return removed + self._prune(resolved.key)

    def unlink_source_generation(
        self,
        source_id: str,
        generation: str,
        *,
        scope: GraphScope | None = None,
    ) -> int:
        source_id = source_id.strip()
        generation = generation.strip()
        if not source_id or not generation:
            return 0
        resolved = self._scope_for_metadata(scope=scope)
        removed = self._remove_evidence(
            resolved.key,
            lambda evidence: (
                evidence.source_id == source_id and evidence.generation == generation
            ),
        )
        return removed + self._prune(resolved.key)

    def _remove_evidence(self, scope_key: str, predicate) -> int:
        removed = 0
        for evidence_map in (self._node_evidence, self._edge_evidence):
            for key, evidence_set in list(evidence_map.items()):
                if key[0] != scope_key:
                    continue
                remaining = {item for item in evidence_set if not predicate(item)}
                removed += len(evidence_set) - len(remaining)
                if remaining:
                    evidence_map[key] = remaining
                else:
                    evidence_map.pop(key, None)
        return removed

    def _prune(self, scope_key: str) -> int:
        removed = 0
        for edge_key in list(self._edge_keys):
            if edge_key[0] == scope_key and not self._edge_evidence.get(edge_key):
                self._edge_keys.discard(edge_key)
                self._edge_evidence.pop(edge_key, None)
        for key in list(self._nodes):
            if key[0] != scope_key or self._entity_supported(key):
                continue
            self._nodes.pop(key, None)
            self._node_evidence.pop(key, None)
            self._adj.pop(key, None)
            removed += 1
        self._rebuild_adjacency()
        return removed

    def _entity_supported(self, key: tuple[str, str]) -> bool:
        if self._node_evidence.get(key):
            return True
        return any(
            edge_key[0] == key[0]
            and (edge_key[1] == key[1] or edge_key[2] == key[1])
            and self._edge_evidence.get(edge_key)
            for edge_key in self._edge_keys
        )

    def _evidence(
        self,
        chunk_id: str,
        metadata: Dict[str, object],
        scope: GraphScope,
        source_id: str | None,
    ) -> _GraphEvidence:
        resolved_source = source_id or str(metadata.get("source_id") or "__legacy__")
        generation = metadata.get(SOURCE_PUBLICATION_GENERATION_METADATA_KEY)
        source_key: RagSourceOperationKey | None = None
        if (
            isinstance(metadata.get("tenant_id"), str)
            and isinstance(metadata.get("source_id"), str)
        ):
            source_key = RagSourceOperationKey(
                tenant_id=scope.tenant_id,
                namespace=scope.namespace,
                workspace_id=scope.workspace_id,
                source_id=str(metadata["source_id"]),
            )
        return _GraphEvidence(
            scope_key=scope.key,
            source_id=resolved_source,
            chunk_id=chunk_id,
            generation=generation if isinstance(generation, str) else None,
            source_key=source_key,
            versioned=SOURCE_PUBLICATION_GENERATION_METADATA_KEY in metadata,
        )

    def _evidence_visible(self, evidence: _GraphEvidence) -> bool:
        return graph_evidence_visible(
            versioned=evidence.versioned,
            generation=evidence.generation,
            source_key=evidence.source_key,
            coordinator=self._source_coordinator,
        )

    def _node_visible(self, key: tuple[str, str]) -> bool:
        evidence = self._node_evidence.get(key)
        return not evidence or any(self._evidence_visible(item) for item in evidence)

    def _has_visible_edge(
        self, source_key: tuple[str, str], target_key: tuple[str, str]
    ) -> bool:
        return any(
            edge_key[0] == source_key[0]
            and (
                (edge_key[1] == source_key[1] and edge_key[2] == target_key[1])
                or (edge_key[1] == target_key[1] and edge_key[2] == source_key[1])
            )
            and any(self._evidence_visible(item) for item in evidence)
            for edge_key, evidence in self._edge_evidence.items()
        )

    def _rebuild_adjacency(self) -> None:
        self._adj.clear()
        for scope_key, source_id, target_id, _relation in self._edge_keys:
            source = (scope_key, source_id)
            target = (scope_key, target_id)
            if source in self._nodes and target in self._nodes:
                self._adj[source].add(target)
                self._adj[target].add(source)

    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        if tenant_id is None and self._bound_scope is not None:
            scope_keys = {self._bound_scope.key}
        elif tenant_id or self._tenant_id:
            tenant = (tenant_id or self._tenant_id or "").strip()
            scope_keys = {
                key
                for key, node in self._nodes
                if str(self._nodes[(key, node)].metadata.get("tenant_id")) == tenant
            }
        else:
            scope_keys = {key[0] for key in self._nodes}
        removed = sum(
            1 for key in list(self._nodes) if key[0] in scope_keys
        )
        for key in list(self._nodes):
            if key[0] in scope_keys:
                self._nodes.pop(key, None)
                self._node_evidence.pop(key, None)
        for edge_key in list(self._edge_keys):
            if edge_key[0] in scope_keys:
                self._edge_keys.discard(edge_key)
                self._edge_evidence.pop(edge_key, None)
        self._rebuild_adjacency()
        return removed
