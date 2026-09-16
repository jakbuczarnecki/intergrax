# © Artur Czarnecki. All rights reserved.

"""Procedural memory capability — recall pipeline over pluggable store (MEM-ENT-8)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticFailureClass,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from intergrax.memory.contracts.memory_security_governance import (
    CanonicalMemoryGovernanceSourceAuthority,
    MemoryGovernanceDenied,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceRecordSnapshot,
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
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.memory_observability_support import emit_procedural_terminal
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_disclosure_governance import (
    evaluate_memory_disclosure,
    memory_security_context_for_recall,
)
from intergrax.memory.memory_specialized_mutation_governance import (
    enforce_specialized_memory_mutation,
    governance_snapshot_from_procedure_record,
    governance_target_for_procedure,
    memory_security_context_for_mutation,
    resolve_governance_source_record_snapshot,
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
    _governance_source_authority: CanonicalMemoryGovernanceSourceAuthority
    _diagnostic_emitter: MemoryDiagnosticEmitter = field(
        default_factory=default_memory_diagnostic_emitter
    )

    def _canonical_source_records(
        self,
        scope: ProceduralMemoryScope,
        record: ProcedureRecord,
        operation: MemoryGovernanceOperation,
    ) -> tuple[MemoryGovernanceRecordSnapshot, ...]:
        preview = record.title[:256] if record.title else None
        snapshot = resolve_governance_source_record_snapshot(
            self._governance_source_authority,
            scope,
            record.source_memory_id,
            record.source_memory_revision,
            operation=operation,
            content_preview=preview,
        )
        return (snapshot,)

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
        proposed_snapshot = governance_snapshot_from_procedure_record(record)
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                proposed_record=proposed_snapshot,
                existing_record=(
                    governance_snapshot_from_procedure_record(existing)
                    if existing is not None
                    else None
                ),
                source_records=self._canonical_source_records(scope, record, operation),
            ),
        )
        stored = self._store.upsert_procedure(scope, record)
        emit_procedural_terminal(
            self._diagnostic_emitter,
            identity=identity,
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            operation=MemoryDiagnosticOperation.REMEMBER,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
            memory_id=stored.source_memory_id,
        )
        return stored

    def recall_procedures(
        self,
        identity: RequestIdentity,
        scope: ProceduralMemoryScope,
        query: ProcedureQuery,
        context: ProcedureRecallContext,
    ) -> ProcedureRecallResult:
        candidates = self._store.query_procedure_candidates(scope, query)
        recall_context = memory_security_context_for_recall(identity, scope)
        disclosed: list[ProcedureRecord] = []
        for record in candidates:
            try:
                sources = self._canonical_source_records(
                    scope, record, MemoryGovernanceOperation.RECALL
                )
            except MemoryGovernanceDenied:
                continue
            if evaluate_memory_disclosure(
                self._security_governance,
                recall_context,
                governance_snapshot_from_procedure_record(record),
                source_records=sources,
            ):
                disclosed.append(record)
        applicable = tuple(
            record
            for record in disclosed
            if self._strategies.applicability.is_applicable(record, context, query=query)
        )
        ranked = self._strategies.ranking.rank(applicable, context)
        bounded = ranked[: query.limit]
        emit_procedural_terminal(
            self._diagnostic_emitter,
            identity=identity,
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            operation=MemoryDiagnosticOperation.RECALL,
            outcome=MemoryDiagnosticOutcome.SUCCESS,
        )
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
        operation = MemoryGovernanceOperation.UPDATE
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                target=governance_target_for_procedure(procedure_id),
                existing_record=governance_snapshot_from_procedure_record(existing),
                proposed_record=governance_snapshot_from_procedure_record(existing),
                source_records=self._canonical_source_records(scope, existing, operation),
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
        operation = MemoryGovernanceOperation.SUPERSEDE
        superseding_snapshot = governance_snapshot_from_procedure_record(
            request.superseding_record
        )
        enforce_specialized_memory_mutation(
            self._security_governance,
            MemoryGovernanceEvaluationRequest(
                context=memory_security_context_for_mutation(identity, scope, operation),
                target=governance_target_for_procedure(request.superseded_procedure_id),
                existing_record=governance_snapshot_from_procedure_record(existing),
                proposed_record=superseding_snapshot,
                source_records=self._canonical_source_records(
                    scope,
                    request.superseding_record,
                    operation,
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
                operation = MemoryGovernanceOperation.DELETE
                enforce_specialized_memory_mutation(
                    self._security_governance,
                    MemoryGovernanceEvaluationRequest(
                        context=memory_security_context_for_mutation(
                            identity, scope, operation
                        ),
                        target=MemoryGovernanceTarget(memory_id=memory_id),
                        existing_record=governance_snapshot_from_procedure_record(existing),
                        source_records=self._canonical_source_records(
                            scope, existing, operation
                        ),
                    ),
                )
        return self._store.delete_by_source_memory(scope, source_memory_id)
