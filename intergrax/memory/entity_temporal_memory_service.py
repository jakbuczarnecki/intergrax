# © Artur Czarnecki. All rights reserved.

"""Entity/temporal memory read orchestration with disclosure governance (MEM-ENT-10C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityEnumerationCapability,
    EntityGraphDisclosureResult,
    EntityMemoryScope,
    EntityRecord,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityRelationResult,
    EntityTemporalMemoryStore,
    order_entity_relations_deterministic,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_disclosure_governance import (
    filter_memory_disclosure_candidates,
    memory_security_context_for_recall,
)
from intergrax.memory.memory_specialized_mutation_governance import (
    governance_snapshot_from_entity_record,
    governance_snapshot_from_entity_relation,
)

__all__ = ["EntityTemporalMemoryService"]


@dataclass(slots=True)
class EntityTemporalMemoryService:
    """Governed read surface over ``EntityTemporalMemoryStore``."""

    _store: EntityTemporalMemoryStore
    _security_governance: MemorySecurityGovernanceService

    def get_entity(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entity_id: str,
        *,
        reference_time: datetime | None = None,
    ) -> EntityRecord | None:
        cache: dict[str, EntityRecord | None] = {}
        return self._resolve_disclosable_entity(
            identity,
            scope,
            entity_id,
            reference_time=reference_time,
            cache=cache,
        )

    def query_relations(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult:
        reference_time = query.as_of
        cache: dict[str, EntityRecord | None] = {}
        if self._store.get_entity(scope, query.entity_id) is not None:
            if (
                self._resolve_disclosable_entity(
                    identity,
                    scope,
                    query.entity_id,
                    reference_time=reference_time,
                    cache=cache,
                )
                is None
            ):
                return EntityRelationResult(relations=())

        raw = self._store.query_relations(scope, query)
        context = memory_security_context_for_recall(
            identity, scope, reference_time=reference_time
        )
        relation_candidates = filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            raw.relations,
            to_snapshot=governance_snapshot_from_entity_relation,
        )
        safe_relations: list[EntityRelationRecord] = []
        for relation in relation_candidates:
            source = self._resolve_disclosable_entity(
                identity,
                scope,
                relation.source_entity_id,
                reference_time=reference_time,
                cache=cache,
            )
            target = self._resolve_disclosable_entity(
                identity,
                scope,
                relation.target_entity_id,
                reference_time=reference_time,
                cache=cache,
            )
            if source is None or target is None:
                continue
            safe_relations.append(relation)
        ordered = order_entity_relations_deterministic(tuple(safe_relations))
        return EntityRelationResult(relations=ordered)

    def list_entities(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
    ) -> tuple[EntityRecord, ...]:
        if not isinstance(self._store, EntityEnumerationCapability):
            return ()
        raw = self._store.list_entities(scope)
        context = memory_security_context_for_recall(identity, scope)
        return filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            raw,
            to_snapshot=governance_snapshot_from_entity_record,
        )

    def disclose_entity_neighbors(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entity_id: str,
        *,
        query: EntityRelationQuery,
    ) -> EntityGraphDisclosureResult:
        relation_result = self.query_relations(identity, scope, query)
        cache: dict[str, EntityRecord | None] = {}
        related_ids: set[str] = set()
        for relation in relation_result.relations:
            if relation.source_entity_id == entity_id:
                related_ids.add(relation.target_entity_id)
            if relation.target_entity_id == entity_id:
                related_ids.add(relation.source_entity_id)
        entities: list[EntityRecord] = []
        for related_id in sorted(related_ids):
            record = self._resolve_disclosable_entity(
                identity,
                scope,
                related_id,
                reference_time=query.as_of,
                cache=cache,
            )
            if record is not None:
                entities.append(record)
        disclosed_ids = {record.entity_id for record in entities}
        structurally_consistent: list[EntityRelationRecord] = []
        for relation in relation_result.relations:
            non_root_endpoints = (
                endpoint
                for endpoint in (
                    relation.source_entity_id,
                    relation.target_entity_id,
                )
                if endpoint != entity_id
            )
            if all(endpoint in disclosed_ids for endpoint in non_root_endpoints):
                structurally_consistent.append(relation)
        return EntityGraphDisclosureResult(
            entities=tuple(entities),
            relations=tuple(structurally_consistent),
        )

    def _resolve_disclosable_entity(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entity_id: str,
        *,
        reference_time: datetime | None,
        cache: dict[str, EntityRecord | None],
    ) -> EntityRecord | None:
        if entity_id in cache:
            return cache[entity_id]
        record = self._store.get_entity(scope, entity_id)
        if record is None:
            cache[entity_id] = None
            return None
        context = memory_security_context_for_recall(
            identity, scope, reference_time=reference_time
        )
        allowed = filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            (record,),
            to_snapshot=governance_snapshot_from_entity_record,
        )
        resolved = allowed[0] if allowed else None
        cache[entity_id] = resolved
        return resolved
