# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Graph store contract for GraphRAG (M-RAG.12) — backend-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Dict, List, Sequence, Set

from intergrax.utils import attribute_access

if TYPE_CHECKING:
    from intergrax.distributed.source_operation import SourceOperationCoordinator


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


@dataclass(frozen=True, slots=True)
class GraphScope:
    """Authoritative graph scope; empty values are canonicalized."""

    tenant_id: str
    namespace: str | None = None
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        tenant_id = self.tenant_id.strip()
        if not tenant_id:
            raise ValueError("graph scope requires tenant_id")
        object.__setattr__(self, "tenant_id", tenant_id)
        object.__setattr__(self, "namespace", _clean_optional(self.namespace))
        object.__setattr__(self, "workspace_id", _clean_optional(self.workspace_id))

    @property
    def key(self) -> str:
        value = "\x1f".join(
            (self.tenant_id, self.namespace or "", self.workspace_id or "")
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def from_object(cls, value: object) -> "GraphScope":
        if isinstance(value, cls):
            return value
        tenant_id = attribute_access.optional(value, "tenant_id", None)
        namespace = attribute_access.optional(value, "namespace", None)
        workspace_id = attribute_access.optional(value, "workspace_id", None)
        if not isinstance(tenant_id, str):
            raise TypeError("scope must expose tenant_id")
        return cls(
            tenant_id=tenant_id,
            namespace=namespace if isinstance(namespace, str) else None,
            workspace_id=workspace_id if isinstance(workspace_id, str) else None,
        )


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class GraphStore(ABC):
    def set_source_operation_coordinator(
        self,
        coordinator: "SourceOperationCoordinator | None",
    ) -> None:
        """Bind publication visibility to the canonical source coordinator.

        Providers that do not implement generation-aware evidence retain their
        existing behavior until they add this optional capability.
        """
        del coordinator

    @property
    def tenant_id(self) -> str | None:
        """Optional tenant namespace for graph isolation (M-RAG.41)."""
        return None

    @property
    def scope(self) -> GraphScope | None:
        """Currently bound full scope, when one has been established."""
        return None

    def bind_scope(self, scope: GraphScope) -> None:
        """Bind the store to one authoritative scope for writes and reads."""
        del scope
        raise NotImplementedError

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
    def unlink_source(self, source_id: str, *, scope: GraphScope | None = None) -> int:
        """Remove only evidence owned by one exact source and scope."""
        raise NotImplementedError

    def unlink_source_generation(
        self,
        source_id: str,
        generation: str,
        *,
        scope: GraphScope | None = None,
    ) -> int:
        """Remove only evidence for one source publication generation and scope."""
        del source_id, generation, scope
        raise NotImplementedError(
            f"{type(self).__name__} does not support generation-specific unlink"
        )

    @abstractmethod
    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        """Remove all RAG graph artifacts, optionally scoped to a tenant (M-RAG.40)."""
        raise NotImplementedError
