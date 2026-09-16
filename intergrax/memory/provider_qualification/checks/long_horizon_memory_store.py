# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.long_horizon_memory import (
    LongHorizonMemoryScope,
    LongHorizonMemoryStore,
    LongHorizonMemoryViolation,
    LongHorizonSummaryRecord,
    MemorySourceRef,
    SummaryNodeKind,
)
from intergrax.memory.contracts.provider_qualification import (
    LongHorizonMemoryStoreQualificationCheck,
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
)
from intergrax.memory.provider_qualification.checks._helpers import failed, passed
from intergrax.memory.provider_qualification.checks._suite import validate_canonical_check_suite

_CAPABILITY = MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


def _summary(
    context: MemoryProviderQualificationContext,
    *,
    suffix: str,
    revision: int,
    content: str,
    covered_from: str = "2025-01-01T00:00:00+00:00",
    covered_until: str = "2025-01-02T00:00:00+00:00",
) -> LongHorizonSummaryRecord:
    summary_id = f"{suffix}-{context.qualification_run_id}"
    return LongHorizonSummaryRecord(
        summary_id=summary_id,
        summary_level=1,
        node_kind=SummaryNodeKind.LEAF,
        revision=revision,
        content=content,
        source_memory_refs=(MemorySourceRef(memory_id=f"src-{suffix}", revision=revision),),
        covered_from=covered_from,
        covered_until=covered_until,
        source_count=1,
        created_at="2025-01-01T00:00:00+00:00",
    )


@dataclass(frozen=True, slots=True)
class LongHorizonTenantIsolationCheck:
    check_id: str = "long_horizon.tenant_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        scope_b = LongHorizonMemoryScope(
            tenant_id=_tenant_b(context),
            user_id=context.user_qualification_id,
        )
        record = _summary(context, suffix="tenant-iso", revision=1, content="tenant-a")
        store.upsert_summary(scope_a, record)
        if store.get_summary(scope_b, record.summary_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonUserIsolationCheck:
    check_id: str = "long_horizon.user_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-a",
        )
        scope_b = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-b",
        )
        record = _summary(context, suffix="user-iso", revision=1, content="user-a")
        store.upsert_summary(scope_a, record)
        sibling = _summary(context, suffix="user-iso", revision=1, content="placeholder")
        if store.get_summary(scope_b, sibling.summary_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonWorkspaceIsolationCheck:
    check_id: str = "long_horizon.workspace_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-a",
        )
        scope_b = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-b",
        )
        record = _summary(context, suffix="ws-iso", revision=1, content="ws-a")
        store.upsert_summary(scope_a, record)
        sibling = _summary(context, suffix="ws-iso", revision=1, content="ws-b-placeholder")
        if store.get_summary(scope_b, sibling.summary_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.WORKSPACE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonStaleRevisionCheck:
    check_id: str = "long_horizon.stale_revision"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-stale",
        )
        current = _summary(context, suffix="stale", revision=3, content="current")
        store.upsert_summary(scope, current)
        stale = _summary(context, suffix="stale", revision=2, content="stale")
        try:
            store.upsert_summary(scope, stale)
        except LongHorizonMemoryViolation:
            stored = store.get_summary(scope, current.summary_id)
            if stored is not None and stored.revision == 3 and stored.content == "current":
                return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)
        stored = store.get_summary(scope, current.summary_id)
        if stored is None or stored.revision != 3 or stored.content != "current":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonSameRevisionIdempotencyCheck:
    check_id: str = "long_horizon.same_revision_idempotency"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-idem",
        )
        record = _summary(context, suffix="idem", revision=2, content="stable")
        first = store.upsert_summary(scope, record)
        second = store.upsert_summary(scope, record)
        if first != second:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonSameRevisionConflictCheck:
    check_id: str = "long_horizon.same_revision_conflict"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-conflict",
        )
        baseline = _summary(context, suffix="conflict", revision=2, content="baseline")
        store.upsert_summary(scope, baseline)
        conflicting = _summary(context, suffix="conflict", revision=2, content="different")
        try:
            store.upsert_summary(scope, conflicting)
        except LongHorizonMemoryViolation:
            return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)
        stored = store.get_summary(scope, baseline.summary_id)
        if stored is not None and stored.content == "baseline":
            return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)
        return failed(
            check_id=self.check_id,
            capability=_CAPABILITY,
            severity=_REQUIRED,
            reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
        )


@dataclass(frozen=True, slots=True)
class LongHorizonHigherRevisionCheck:
    check_id: str = "long_horizon.higher_revision"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-higher",
        )
        initial = _summary(context, suffix="higher", revision=1, content="one")
        store.upsert_summary(scope, initial)
        updated = _summary(context, suffix="higher", revision=2, content="two")
        store.upsert_summary(scope, updated)
        stored = store.get_summary(scope, initial.summary_id)
        if stored is None or stored.revision != 2 or stored.content != "two":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class LongHorizonSourceFidelityCheck:
    check_id: str = "long_horizon.source_fidelity"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: LongHorizonMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = LongHorizonMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-fidelity",
            workspace_id=context.workspace_qualification_id,
        )
        record = _summary(
            context,
            suffix="fidelity",
            revision=4,
            content="payload",
            covered_from="2025-03-01T00:00:00+00:00",
            covered_until="2025-03-10T00:00:00+00:00",
        )
        store.upsert_summary(scope, record)
        stored = store.get_summary(scope, record.summary_id)
        if stored is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.SOURCE_FIDELITY_FAILURE,
            )
        if not stored.source_memory_refs or stored.source_memory_refs[0].revision != 4:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.SOURCE_FIDELITY_FAILURE,
            )
        if stored.covered_from != record.covered_from or stored.covered_until != record.covered_until:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.SOURCE_FIDELITY_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


LONG_HORIZON_MEMORY_STORE_CHECKS: tuple[LongHorizonMemoryStoreQualificationCheck, ...] = (
    LongHorizonTenantIsolationCheck(),
    LongHorizonUserIsolationCheck(),
    LongHorizonWorkspaceIsolationCheck(),
    LongHorizonStaleRevisionCheck(),
    LongHorizonSameRevisionIdempotencyCheck(),
    LongHorizonSameRevisionConflictCheck(),
    LongHorizonHigherRevisionCheck(),
    LongHorizonSourceFidelityCheck(),
)

validate_canonical_check_suite(
    LONG_HORIZON_MEMORY_STORE_CHECKS,
    capability=_CAPABILITY,
    check_id_of=lambda item: item.check_id,
    capability_of=lambda item: item.capability,
    severity_of=lambda item: item.severity,
)


def default_long_horizon_checks() -> tuple[LongHorizonMemoryStoreQualificationCheck, ...]:
    return LONG_HORIZON_MEMORY_STORE_CHECKS
