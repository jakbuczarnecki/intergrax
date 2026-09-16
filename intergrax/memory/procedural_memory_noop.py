# © Artur Czarnecki. All rights reserved.

"""Explicit no-op procedural memory backend (MEM-ENT-8)."""

from __future__ import annotations

from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureQuery,
    ProcedureRecord,
    ProcedureSupersessionRequest,
)


class NoOpProceduralMemoryStore:
    """Configured disabled backend; operations are inert but typed."""

    def upsert_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord:
        return record

    def get_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        return None

    def query_procedure_candidates(
        self,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
    ) -> tuple[ProcedureRecord, ...]:
        return ()

    def deprecate_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        return None

    def apply_supersession(
        self,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]:
        return request.superseding_record, request.superseding_record

    def delete_by_source_memory(self, scope: ProceduralMemoryScope, source_memory_id: str) -> int:
        return 0
