# © Artur Czarnecki. All rights reserved.

"""Procedural memory capability — recall pipeline over pluggable store (MEM-ENT-8)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.procedural_memory import (
    DefaultProcedureApplicabilityStrategy,
    DefaultProcedureRankingStrategy,
    ProcedureApplicabilityStrategy,
    ProcedureMemoryCapability,
    ProceduralMemoryScope,
    ProcedureMemoryStore,
    ProcedureQuery,
    ProcedureRankingStrategy,
    ProcedureRecallContext,
    ProcedureRecallResult,
    ProcedureRecord,
    ProcedureSupersessionRequest,
)

__all__ = ["ProceduralMemoryService", "ProceduralMemoryStrategySet"]


@dataclass(frozen=True, slots=True)
class ProceduralMemoryStrategySet:
    applicability: ProcedureApplicabilityStrategy
    ranking: ProcedureRankingStrategy


def build_default_procedural_memory_strategies() -> ProceduralMemoryStrategySet:
    return ProceduralMemoryStrategySet(
        applicability=DefaultProcedureApplicabilityStrategy(),
        ranking=DefaultProcedureRankingStrategy(),
    )


@dataclass(slots=True)
class ProceduralMemoryService:
    """Default ``ProcedureMemoryCapability`` backed by ``ProcedureMemoryStore``."""

    _store: ProcedureMemoryStore
    _strategies: ProceduralMemoryStrategySet

    def remember_procedure(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord:
        """Upsert a canonical procedural projection (requires source memory linkage)."""
        return self._store.upsert_procedure(scope, record)

    def recall_procedures(
        self,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
        context: ProcedureRecallContext,
    ) -> ProcedureRecallResult:
        candidates = self._store.query_procedure_candidates(scope, query)
        applicable = tuple(
            record
            for record in candidates
            if self._strategies.applicability.is_applicable(record, context, query=query)
        )
        ranked = self._strategies.ranking.rank(applicable, context)
        bounded = ranked[: query.limit]
        return ProcedureRecallResult(procedures=bounded)

    def deprecate_procedure(
        self,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        return self._store.deprecate_procedure(scope, procedure_id)

    def supersede_procedure(
        self,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]:
        return self._store.apply_supersession(scope, request)

    def delete_projection_by_source_memory(
        self,
        scope: ProceduralMemoryScope,
        source_memory_id: str,
    ) -> int:
        return self._store.delete_by_source_memory(scope, source_memory_id)
