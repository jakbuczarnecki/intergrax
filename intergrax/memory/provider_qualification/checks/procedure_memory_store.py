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
    ProcedureMemoryViolation,
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
from intergrax.memory.provider_qualification.checks._suite import validate_canonical_check_suite

_CAPABILITY = MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


def _procedure_record(
    scope: ProceduralMemoryScope,
    context: MemoryProviderQualificationContext,
    *,
    suffix: str,
    title: str,
    source_revision: int,
) -> ProcedureRecord:
    memory_id = f"{suffix}-{context.qualification_run_id}"
    procedure_id = procedure_id_for_source_memory(scope, memory_id)
    return ProcedureRecord(
        procedure_id=procedure_id,
        procedure_type=ProcedureTypeRef("runbook"),
        title=title,
        source_memory_id=memory_id,
        source_memory_revision=source_revision,
        revision=source_revision,
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
        scope_a = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        scope_b = ProceduralMemoryScope(
            tenant_id=_tenant_b(context),
            user_id=context.user_qualification_id,
        )
        record = _procedure_record(
            scope_a,
            context,
            suffix="tenant-iso",
            title="tenant-a",
            source_revision=1,
        )
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


@dataclass(frozen=True, slots=True)
class ProcedureMemoryUserIsolationCheck:
    check_id: str = "procedure_memory.user_isolation"

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
        scope_a = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-a",
        )
        scope_b = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-b",
        )
        memory_id = f"user-iso-{context.qualification_run_id}"
        record = _procedure_record(
            scope_a,
            context,
            suffix="user-iso",
            title="user-a",
            source_revision=1,
        )
        sibling_id = procedure_id_for_source_memory(scope_b, memory_id)
        store.upsert_procedure(scope_a, record)
        if store.get_procedure(scope_b, sibling_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemoryWorkspaceIsolationCheck:
    check_id: str = "procedure_memory.workspace_isolation"

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
        scope_a = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-a",
        )
        scope_b = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-b",
        )
        memory_id = f"ws-iso-{context.qualification_run_id}"
        record = _procedure_record(
            scope_a,
            context,
            suffix="ws-iso",
            title="ws-a",
            source_revision=1,
        )
        sibling_id = procedure_id_for_source_memory(scope_b, memory_id)
        store.upsert_procedure(scope_a, record)
        if store.get_procedure(scope_b, sibling_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.WORKSPACE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemoryStaleRevisionCheck:
    check_id: str = "procedure_memory.stale_source_revision"

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
        scope = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-stale",
        )
        current = _procedure_record(
            scope,
            context,
            suffix="stale-rev",
            title="rev-four",
            source_revision=4,
        )
        store.upsert_procedure(scope, current)
        stale = _procedure_record(
            scope,
            context,
            suffix="stale-rev",
            title="stale",
            source_revision=3,
        )
        result = store.upsert_procedure(scope, stale)
        stored = store.get_procedure(scope, current.procedure_id)
        if stored is None or stored.source_memory_revision != 4 or stored.title != "rev-four":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        if result.source_memory_revision != stored.source_memory_revision:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemorySameRevisionIdempotencyCheck:
    check_id: str = "procedure_memory.same_revision_idempotency"

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
        scope = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-idem",
        )
        record = _procedure_record(
            scope,
            context,
            suffix="same-rev",
            title="stable",
            source_revision=2,
        )
        first = store.upsert_procedure(scope, record)
        second = store.upsert_procedure(scope, record)
        if first != second:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemorySameRevisionConflictCheck:
    check_id: str = "procedure_memory.same_revision_conflict"

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
        scope = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-conflict",
        )
        baseline = _procedure_record(
            scope,
            context,
            suffix="conflict",
            title="baseline",
            source_revision=2,
        )
        store.upsert_procedure(scope, baseline)
        conflicting = _procedure_record(
            scope,
            context,
            suffix="conflict",
            title="conflicting-title",
            source_revision=2,
        )
        try:
            store.upsert_procedure(scope, conflicting)
        except ProcedureMemoryViolation:
            return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)
        stored = store.get_procedure(scope, baseline.procedure_id)
        if stored is not None and stored.title == "baseline":
            return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)
        return failed(
            check_id=self.check_id,
            capability=_CAPABILITY,
            severity=_REQUIRED,
            reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
        )


@dataclass(frozen=True, slots=True)
class ProcedureMemoryHigherRevisionCheck:
    check_id: str = "procedure_memory.higher_source_revision"

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
        scope = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-higher",
        )
        initial = _procedure_record(
            scope,
            context,
            suffix="higher",
            title="one",
            source_revision=1,
        )
        store.upsert_procedure(scope, initial)
        updated = _procedure_record(
            scope,
            context,
            suffix="higher",
            title="two",
            source_revision=2,
        )
        store.upsert_procedure(scope, updated)
        stored = store.get_procedure(scope, initial.procedure_id)
        if stored is None or stored.source_memory_revision != 2 or stored.title != "two":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemoryDeleteScopeCheck:
    check_id: str = "procedure_memory.delete_scope"

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
        scope_a = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-a",
        )
        scope_b = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-b",
        )
        record_a = _procedure_record(
            scope_a,
            context,
            suffix="del-a",
            title="delete-a",
            source_revision=1,
        )
        record_b = _procedure_record(
            scope_b,
            context,
            suffix="del-b",
            title="delete-b",
            source_revision=1,
        )
        store.upsert_procedure(scope_a, record_a)
        store.upsert_procedure(scope_b, record_b)
        store.delete_by_source_memory(scope_a, record_a.source_memory_id or "")
        if store.get_procedure(scope_b, record_b.procedure_id) is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.DELETE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class ProcedureMemoryDeleteIdempotencyCheck:
    check_id: str = "procedure_memory.delete_idempotency"

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
        scope = ProceduralMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-idem",
        )
        record = _procedure_record(
            scope,
            context,
            suffix="del-idem",
            title="gone",
            source_revision=1,
        )
        store.upsert_procedure(scope, record)
        memory_id = record.source_memory_id or ""
        first = store.delete_by_source_memory(scope, memory_id)
        second = store.delete_by_source_memory(scope, memory_id)
        missing = store.delete_by_source_memory(scope, "missing-proc-memory")
        if first != 1 or second != 0 or missing != 0:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.INVALID_FAILURE_BEHAVIOR,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


PROCEDURE_MEMORY_STORE_CHECKS: tuple[ProcedureMemoryStoreQualificationCheck, ...] = (
    ProcedureMemoryTenantIsolationCheck(),
    ProcedureMemoryUserIsolationCheck(),
    ProcedureMemoryWorkspaceIsolationCheck(),
    ProcedureMemoryStaleRevisionCheck(),
    ProcedureMemorySameRevisionIdempotencyCheck(),
    ProcedureMemorySameRevisionConflictCheck(),
    ProcedureMemoryHigherRevisionCheck(),
    ProcedureMemoryDeleteScopeCheck(),
    ProcedureMemoryDeleteIdempotencyCheck(),
)

validate_canonical_check_suite(
    PROCEDURE_MEMORY_STORE_CHECKS,
    capability=_CAPABILITY,
    check_id_of=lambda item: item.check_id,
    capability_of=lambda item: item.capability,
    severity_of=lambda item: item.severity,
)


def default_procedure_checks() -> tuple[ProcedureMemoryStoreQualificationCheck, ...]:
    return PROCEDURE_MEMORY_STORE_CHECKS
