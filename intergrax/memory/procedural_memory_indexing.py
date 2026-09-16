# © Artur Czarnecki. All rights reserved.

"""Index canonical memory entries into procedural projections (MEM-ENT-8)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureMemoryStore,
    ProcedureRecord,
    ProcedureStatus,
    ProcedureTypeRef,
    procedure_id_for_source_memory,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


class DefaultProceduralMemoryIndexer:
    """Materializes minimal procedural projection rows from canonical LTM entries."""

    def __init__(self, store: ProcedureMemoryStore) -> None:
        self._store = store

    def index_memory_entry(
        self,
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
        return self._store.upsert_procedure(scope, record)

    def remove_memory_entry(self, scope: ProceduralMemoryScope, memory_entry_id: str) -> int:
        return self._store.delete_by_source_memory(scope, memory_entry_id)
