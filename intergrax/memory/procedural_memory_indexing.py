# © Artur Czarnecki. All rights reserved.

"""Index canonical memory entries into procedural projections (MEM-ENT-8)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceTarget,
)
from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureMemoryStore,
    ProcedureRecord,
    ProcedureStatus,
    ProcedureTypeRef,
    procedure_id_for_source_memory,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    governance_snapshot_from_procedure_record,
    governance_source_snapshot_from_user_entry,
    memory_security_context_for_mutation,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


class DefaultProceduralMemoryIndexer:
    """Materializes minimal procedural projection rows from canonical LTM entries."""

    def __init__(
        self,
        store: ProcedureMemoryStore,
        *,
        security_governance: MemorySecurityGovernanceService,
    ) -> None:
        self._store = store
        self._security_governance = security_governance

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> ProcedureRecord | None:
        if entry.kind is not MemoryKind.PROCEDURAL:
            return None
        procedure_id = procedure_id_for_source_memory(scope, entry.entry_id)
        title = (entry.title or entry.content or entry.entry_id).strip() or entry.entry_id
        record = ProcedureRecord(
            procedure_id=procedure_id,
            procedure_type=ProcedureTypeRef("canonical_memory_projection"),
            title=title[:512],
            source_memory_id=entry.entry_id,
            source_memory_revision=max(1, int(entry.revision or 1)),
            revision=max(1, int(entry.revision or 1)),
            status=ProcedureStatus.ACTIVE if not entry.deleted else ProcedureStatus.DISABLED,
            steps=(),
            provenance=entry.provenance,
            trust=entry.trust,
            governance=entry.governance,
            evidence_refs=entry.evidence_refs,
            created_at=entry.created_at or "",
            updated_at=entry.updated_at,
        )
        operation = (
            MemoryGovernanceOperation.DELETE
            if entry.deleted
            else MemoryGovernanceOperation.PROJECT
        )
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                proposed_record=governance_snapshot_from_procedure_record(record),
                source_records=(governance_source_snapshot_from_user_entry(entry),),
                target=MemoryGovernanceTarget(memory_id=entry.entry_id) if entry.deleted else None,
            ),
        )
        return self._store.upsert_procedure(scope, record)

    def remove_memory_entry(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        memory_entry_id: str,
    ) -> int:
        memory_id = (memory_entry_id or "").strip()
        if memory_id:
            procedure_id = procedure_id_for_source_memory(scope, memory_id)
            existing = self._store.get_procedure(scope, procedure_id)
            if existing is not None:
                enforce_specialized_memory_mutation(
                    self._security_governance,
                    MemoryGovernanceEvaluationRequest(
                        context=memory_security_context_for_mutation(
                            identity, scope, MemoryGovernanceOperation.DELETE
                        ),
                        target=MemoryGovernanceTarget(memory_id=memory_id),
                        existing_record=governance_snapshot_from_procedure_record(existing),
                    ),
                )
        return self._store.delete_by_source_memory(scope, memory_entry_id)
