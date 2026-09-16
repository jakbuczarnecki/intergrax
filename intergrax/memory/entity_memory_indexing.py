# © Artur Czarnecki. All rights reserved.

"""Projection indexer from canonical memory records into entity/temporal store (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    RelationTypeRef,
    entity_memory_entity_id_for_entry,
    entity_memory_relation_id_for_has_memory,
    entity_memory_user_entity_id,
)
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    governance_snapshot_from_entity_record,
    governance_source_snapshot_from_user_entry,
    memory_security_context_for_mutation,
)


def _source_revision_stale(
    stored_revision: int | None,
    incoming_revision: int,
) -> bool:
    return stored_revision is not None and incoming_revision < stored_revision


def _entity_payload_unchanged(existing: EntityRecord, incoming: EntityRecord) -> bool:
    return (
        existing.canonical_name == incoming.canonical_name
        and existing.entity_type == incoming.entity_type
        and existing.created_at == incoming.created_at
        and existing.updated_at == incoming.updated_at
        and existing.aliases == incoming.aliases
        and existing.provenance == incoming.provenance
        and existing.trust == incoming.trust
        and existing.governance == incoming.governance
        and existing.evidence_refs == incoming.evidence_refs
        and existing.source_memory_id == incoming.source_memory_id
        and existing.source_memory_revision == incoming.source_memory_revision
        and existing.revision == incoming.revision
    )


class DefaultEntityMemoryIndexer:
    """Derived projection indexer: LTM entries → entity graph records."""

    def __init__(
        self,
        store: EntityTemporalMemoryStore,
        *,
        security_governance: MemorySecurityGovernanceService,
    ) -> None:
        self._store = store
        self._security_governance = security_governance

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if entry.deleted:
            self._enforce_delete_projection(identity, scope, entry)
            self._store.delete_by_source_memory(scope, entry.entry_id)
            return
        content = (entry.content or "").strip()
        if not content:
            return

        user_id = (scope.user_id or "").strip()
        if not user_id:
            raise ValueError("scope.user_id is required for entity memory indexing")

        entity_type = (
            entry.kind.value if isinstance(entry.kind, MemoryKind) else str(entry.kind)
        )
        memory_entity_id = entity_memory_entity_id_for_entry(scope, entry.entry_id)
        existing = self._store.get_entity(scope, memory_entity_id)
        if existing is not None and _source_revision_stale(
            existing.source_memory_revision,
            entry.revision,
        ):
            return

        incoming_entity = EntityRecord(
            entity_id=memory_entity_id,
            entity_type=EntityTypeRef(entity_type),
            canonical_name=content[:120],
            revision=entry.revision,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            provenance=entry.provenance,
            trust=entry.trust,
            governance=entry.governance,
            evidence_refs=entry.evidence_refs,
            source_memory_id=entry.entry_id,
            source_memory_revision=entry.revision,
        )
        if existing is not None and _entity_payload_unchanged(existing, incoming_entity):
            return

        operation = (
            MemoryGovernanceOperation.UPDATE
            if existing is not None
            else MemoryGovernanceOperation.PROJECT
        )
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                proposed_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
                existing_record=(
                    governance_snapshot_from_entity_record(existing)
                    if existing is not None
                    else None
                ),
                source_records=(governance_source_snapshot_from_user_entry(entry),),
            ),
        )

        self._store.upsert_entity(scope, incoming_entity)

        user_entity_id = entity_memory_user_entity_id(scope)
        user_existing = self._store.get_entity(scope, user_entity_id)
        self._store.upsert_entity(
            scope,
            EntityRecord(
                entity_id=user_entity_id,
                entity_type=EntityTypeRef("user"),
                canonical_name=user_id,
                revision=user_existing.revision if user_existing else 1,
                provenance=entry.provenance,
                trust=entry.trust,
                governance=entry.governance,
            ),
        )

        relation_id = entity_memory_relation_id_for_has_memory(scope, entry.entry_id)
        self._store.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id=relation_id,
                source_entity_id=user_entity_id,
                target_entity_id=memory_entity_id,
                relation_type=RelationTypeRef("has_memory"),
                revision=entry.revision,
                valid_from=entry.valid_from,
                valid_until=entry.valid_until,
                provenance=entry.provenance,
                trust=entry.trust,
                governance=entry.governance,
                evidence_refs=entry.evidence_refs,
                lineage=entry.lineage,
                source_memory_id=entry.entry_id,
                source_memory_revision=entry.revision,
            ),
        )

    def _enforce_delete_projection(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(
                    identity, scope, MemoryGovernanceOperation.DELETE
                ),
                target=MemoryGovernanceTarget(memory_id=entry.entry_id),
                existing_record=MemoryGovernanceRecordSnapshot.from_user_profile_entry(entry),
            ),
        )

    def remove_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        memory_entry_id: str,
    ) -> None:
        memory_id = (memory_entry_id or "").strip()
        if memory_id:
            memory_entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
            existing = self._store.get_entity(scope, memory_entity_id)
            if existing is not None:
                enforce_specialized_memory_mutation(
                    self._security_governance,
                    MemoryGovernanceEvaluationRequest(
                        context=memory_security_context_for_mutation(
                            identity, scope, MemoryGovernanceOperation.DELETE
                        ),
                        target=MemoryGovernanceTarget(memory_id=memory_id),
                        existing_record=governance_snapshot_from_entity_record(existing),
                    ),
                )
        self._store.delete_by_source_memory(scope, memory_entry_id)
