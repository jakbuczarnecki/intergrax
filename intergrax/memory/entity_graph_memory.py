# © Artur Czarnecki. All rights reserved.

"""Legacy entity graph DTOs and facade over canonical entity/temporal store (MEM-ENT-7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityEnumerationCapability,
    EntityMemoryScope,
    EntityRecord,
    EntityRelationDirection,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    RelationTypeRef,
)
from intergrax.utils.time_provider import SystemTimeProvider, TimeProvider

_LEGACY_TENANT_ID = "legacy-default"


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
    Backward-compatible facade for user-scoped entity graph memory.

    New code should depend on ``EntityTemporalMemoryStore`` directly.
    """

    def __init__(
        self,
        backend: EntityTemporalMemoryStore,
        *,
        tenant_id: str = _LEGACY_TENANT_ID,
        time_provider: TimeProvider | None = None,
    ) -> None:
        self._backend = backend
        self._tenant_id = tenant_id
        self._time_provider = time_provider or SystemTimeProvider()

    @property
    def entity_temporal_store(self) -> EntityTemporalMemoryStore:
        return self._backend

    def upsert_node(self, node: EntityNode) -> None:
        scope = EntityMemoryScope(tenant_id=self._tenant_id)
        self._backend.upsert_entity(
            scope,
            EntityRecord(
                entity_id=node.entity_id,
                entity_type=EntityTypeRef(node.entity_type),
                canonical_name=node.label,
                revision=1,
            ),
        )

    def add_edge(self, edge: EntityEdge) -> None:
        scope = EntityMemoryScope(tenant_id=self._tenant_id)
        relation_id = f"rel:legacy:{edge.source_id}:{edge.target_id}:{edge.relation}"
        self._backend.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id=relation_id,
                source_entity_id=edge.source_id,
                target_entity_id=edge.target_id,
                relation_type=RelationTypeRef(edge.relation),
                revision=1,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            ),
        )

    def neighbors(
        self,
        entity_id: str,
        *,
        as_of: datetime | None = None,
    ) -> List[EntityNode]:
        scope = EntityMemoryScope(tenant_id=self._tenant_id)
        reference = as_of if as_of is not None else self._time_provider.utc_now()
        result = self._backend.query_relations(
            scope,
            EntityRelationQuery(
                entity_id=entity_id,
                direction=EntityRelationDirection.BOTH,
                as_of=reference,
                limit=500,
            ),
        )
        related_ids: set[str] = set()
        for relation in result.relations:
            if relation.source_entity_id == entity_id:
                related_ids.add(relation.target_entity_id)
            if relation.target_entity_id == entity_id:
                related_ids.add(relation.source_entity_id)
        nodes: List[EntityNode] = []
        for related_id in sorted(related_ids):
            record = self._backend.get_entity(scope, related_id)
            if record is None:
                continue
            nodes.append(
                EntityNode(
                    entity_id=record.entity_id,
                    label=record.canonical_name,
                    entity_type=record.entity_type.value,
                )
            )
        return nodes

    def list_nodes(self) -> List[EntityNode]:
        if not isinstance(self._backend, EntityEnumerationCapability):
            return []
        scope = EntityMemoryScope(tenant_id=self._tenant_id)
        nodes: List[EntityNode] = []
        for record in self._backend.list_entities(scope):
            nodes.append(
                EntityNode(
                    entity_id=record.entity_id,
                    label=record.canonical_name,
                    entity_type=record.entity_type.value,
                )
            )
        return nodes
