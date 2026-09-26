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
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticFailureClass,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
    MemoryGovernanceTarget,
)
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.memory_observability_support import (
    emit_entity_projection_terminal,
    governance_failure_class,
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
        diagnostic_emitter: MemoryDiagnosticEmitter | None = None,
    ) -> None:
        self._store = store
        self._security_governance = security_governance
        self._diagnostic_emitter = (
            diagnostic_emitter
            if diagnostic_emitter is not None
            else default_memory_diagnostic_emitter()
        )

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if entry.deleted:
            self._project_delete_from_entry(identity, scope, entry)
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
        try:
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
            self._upsert_projection_graph(
                scope,
                entry,
                incoming_entity,
                memory_entity_id,
                user_id,
            )
        except MemoryGovernanceDenied as exc:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_WRITE,
                outcome=MemoryDiagnosticOutcome.DENIED,
                memory_id=entry.entry_id,
                revision=entry.revision,
                projection_id=memory_entity_id,
                failure_class=governance_failure_class(exc.decision),
            )
            raise
        except Exception:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_WRITE,
                outcome=MemoryDiagnosticOutcome.FAILED,
                memory_id=entry.entry_id,
                revision=entry.revision,
                projection_id=memory_entity_id,
                failure_class=MemoryDiagnosticFailureClass.STORE,
            )
            raise
        self._emit_projection_terminal(
            scope,
            operation=MemoryDiagnosticOperation.PROJECTION_WRITE,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
            memory_id=entry.entry_id,
            revision=entry.revision,
            projection_id=memory_entity_id,
        )

    def _upsert_projection_graph(
        self,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
        incoming_entity: EntityRecord,
        memory_entity_id: str,
        user_id: str,
    ) -> None:
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

    def _project_delete_from_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> None:
        memory_entity_id = entity_memory_entity_id_for_entry(scope, entry.entry_id)
        try:
            self._enforce_delete_projection(identity, scope, entry)
            self._store.delete_by_source_memory(scope, entry.entry_id)
        except MemoryGovernanceDenied as exc:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
                outcome=MemoryDiagnosticOutcome.DENIED,
                memory_id=entry.entry_id,
                revision=entry.revision,
                projection_id=memory_entity_id,
                failure_class=governance_failure_class(exc.decision),
            )
            raise
        except Exception:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
                outcome=MemoryDiagnosticOutcome.FAILED,
                memory_id=entry.entry_id,
                revision=entry.revision,
                projection_id=memory_entity_id,
                failure_class=MemoryDiagnosticFailureClass.STORE,
            )
            raise
        self._emit_projection_terminal(
            scope,
            operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
            memory_id=entry.entry_id,
            revision=entry.revision,
            projection_id=memory_entity_id,
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
        memory_entity_id = (
            entity_memory_entity_id_for_entry(scope, memory_id) if memory_id else None
        )
        try:
            if memory_id and memory_entity_id is not None:
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
        except MemoryGovernanceDenied as exc:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
                outcome=MemoryDiagnosticOutcome.DENIED,
                memory_id=memory_id or None,
                projection_id=memory_entity_id,
                failure_class=governance_failure_class(exc.decision),
            )
            raise
        except Exception:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
                outcome=MemoryDiagnosticOutcome.FAILED,
                memory_id=memory_id or None,
                projection_id=memory_entity_id,
                failure_class=MemoryDiagnosticFailureClass.STORE,
            )
            raise
        if memory_id:
            self._emit_projection_terminal(
                scope,
                operation=MemoryDiagnosticOperation.PROJECTION_DELETE,
                outcome=MemoryDiagnosticOutcome.SUCCESS,
                memory_id=memory_id,
                projection_id=memory_entity_id,
            )

    def _emit_projection_terminal(
        self,
        scope: EntityMemoryScope,
        *,
        operation: MemoryDiagnosticOperation,
        outcome: MemoryDiagnosticOutcome,
        memory_id: str | None = None,
        revision: int | None = None,
        projection_id: str | None = None,
        failure_class: MemoryDiagnosticFailureClass | None = None,
    ) -> None:
        emit_entity_projection_terminal(
            self._diagnostic_emitter,
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            operation=operation,
            outcome=outcome,
            memory_id=memory_id,
            revision=revision,
            projection_id=projection_id,
            failure_class=failure_class,
        )
