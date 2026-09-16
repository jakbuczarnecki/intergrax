# © Artur Czarnecki. All rights reserved.

"""Procedural memory capability — recall pipeline over pluggable store (MEM-ENT-8)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceTarget,
)
from intergrax.memory.contracts.procedural_memory import (
    DefaultProcedureApplicabilityStrategy,
    DefaultProcedureRankingStrategy,
    ProcedureApplicabilityStrategy,
    ProcedureMemoryCapability,
    ProcedureMemoryViolation,
    ProceduralMemoryScope,
    ProcedureMemoryStore,
    ProcedureQuery,
    ProcedureRankingStrategy,
    ProcedureRecallContext,
    ProcedureRecallResult,
    ProcedureRecord,
    ProcedureSupersessionRequest,
    procedure_id_for_source_memory,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    governance_snapshot_from_procedure_record,
    governance_target_for_procedure,
    memory_security_context_for_mutation,
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
    _security_governance: MemorySecurityGovernanceService

    def remember_procedure(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
    ) -> ProcedureRecord:
        """Upsert a canonical procedural projection (requires source memory linkage)."""
        existing = self._store.get_procedure(scope, record.procedure_id)
        operation = (
            MemoryGovernanceOperation.UPDATE
            if existing is not None
            else MemoryGovernanceOperation.REMEMBER
        )
        source_snapshot = governance_snapshot_from_procedure_record(record)
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                proposed_record=source_snapshot,
                existing_record=(
                    governance_snapshot_from_procedure_record(existing)
                    if existing is not None
                    else None
                ),
                source_records=(source_snapshot,),
            ),
        )
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
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        procedure_id: str,
    ) -> ProcedureRecord | None:
        existing = self._store.get_procedure(scope, procedure_id)
        if existing is None:
            return None
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(
                    identity, scope, MemoryGovernanceOperation.UPDATE
                ),
                target=governance_target_for_procedure(procedure_id),
                existing_record=governance_snapshot_from_procedure_record(existing),
                proposed_record=governance_snapshot_from_procedure_record(existing),
                source_records=(governance_snapshot_from_procedure_record(existing),),
            ),
        )
        return self._store.deprecate_procedure(scope, procedure_id)

    def supersede_procedure(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        request: ProcedureSupersessionRequest,
    ) -> tuple[ProcedureRecord, ProcedureRecord]:
        existing = self._store.get_procedure(scope, request.superseded_procedure_id)
        if existing is None:
            raise ProcedureMemoryViolation("superseded procedure not found")
        superseding_snapshot = governance_snapshot_from_procedure_record(
            request.superseding_record
        )
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(
                    identity, scope, MemoryGovernanceOperation.SUPERSEDE
                ),
                target=governance_target_for_procedure(request.superseded_procedure_id),
                existing_record=governance_snapshot_from_procedure_record(existing),
                proposed_record=superseding_snapshot,
                source_records=(
                    governance_snapshot_from_procedure_record(existing),
                    superseding_snapshot,
                ),
            ),
        )
        return self._store.apply_supersession(scope, request)

    def delete_projection_by_source_memory(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        source_memory_id: str,
    ) -> int:
        memory_id = (source_memory_id or "").strip()
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
        return self._store.delete_by_source_memory(scope, source_memory_id)
