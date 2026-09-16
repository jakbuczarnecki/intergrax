# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureActionKind,
    ProcedureApplicability,
    ProcedureMemoryStore,
    ProcedureOutcomeEvidence,
    ProcedureRecord,
    ProcedureStatus,
    ProcedureStep,
    ProcedureToolReference,
    ProcedureTypeRef,
    ProceduralMemoryScope,
    procedure_id_for_source_memory,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    ProcedureMemoryStoreQualificationCheck,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
)
from intergrax.memory.provider_qualification.checks._helpers import failed, passed

_CAPABILITY = MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


def _procedure_record(scope: ProceduralMemoryScope, memory_id: str) -> ProcedureRecord:
    procedure_id = procedure_id_for_source_memory(scope, memory_id)
    return ProcedureRecord(
        procedure_id=procedure_id,
        procedure_type=ProcedureTypeRef("runbook"),
        title=f"qual-{memory_id}",
        source_memory_id=memory_id,
        source_memory_revision=1,
        revision=1,
        status=ProcedureStatus.ACTIVE,
        steps=(
            ProcedureStep(
                step_id="s1",
                position=0,
                action_kind=ProcedureActionKind.TOOL_ACTION,
                instruction="step",
                tool_reference=ProcedureToolReference(tool_capability_id="cap"),
            ),
        ),
        applicability=ProcedureApplicability(),
        outcome_evidence=ProcedureOutcomeEvidence(quality_score=1.0),
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        created_at="2025-01-01T00:00:00+00:00",
    )


@dataclass(frozen=True, slots=True)
class ProcedureMemoryTenantIsolationCheck:
    check_id: str = "procedure_memory.tenant_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: ProcedureMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        memory_id = f"proc-{context.qualification_run_id}"
        scope_a = ProceduralMemoryScope(tenant_id=_tenant_a(context), user_id=context.user_qualification_id)
        scope_b = ProceduralMemoryScope(tenant_id=_tenant_b(context), user_id=context.user_qualification_id)
        record = _procedure_record(scope_a, memory_id)
        store.upsert_procedure(scope_a, record)
        leaked = store.get_procedure(scope_b, record.procedure_id)
        if leaked is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


PROCEDURE_MEMORY_STORE_CHECKS: tuple[ProcedureMemoryStoreQualificationCheck, ...] = (
    ProcedureMemoryTenantIsolationCheck(),
)


def default_procedure_checks() -> tuple[ProcedureMemoryStoreQualificationCheck, ...]:
    return PROCEDURE_MEMORY_STORE_CHECKS
