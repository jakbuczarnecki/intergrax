# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.long_horizon_memory import (
    LongHorizonMemoryScope,
    LongHorizonMemoryStore,
    LongHorizonSummaryRecord,
    MemorySourceRef,
    SummaryNodeKind,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderQualificationCheck,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
)
from intergrax.memory.provider_qualification.checks._helpers import failed, passed

_CAPABILITY = MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


def _summary(summary_id: str) -> LongHorizonSummaryRecord:
    return LongHorizonSummaryRecord(
        summary_id=summary_id,
        summary_level=1,
        node_kind=SummaryNodeKind.LEAF,
        revision=1,
        content="qualification leaf",
        source_memory_refs=(MemorySourceRef(memory_id="src-1", revision=1),),
        covered_from="2025-01-01T00:00:00+00:00",
        covered_until="2025-01-02T00:00:00+00:00",
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
        instance: object,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        if not isinstance(store, LongHorizonMemoryStore):
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        summary_id = f"lh-{context.qualification_run_id}"
        scope_a = LongHorizonMemoryScope(tenant_id=_tenant_a(context), user_id=context.user_qualification_id)
        scope_b = LongHorizonMemoryScope(tenant_id=_tenant_b(context), user_id=context.user_qualification_id)
        store.upsert_summary(scope_a, _summary(summary_id))
        leaked = store.get_summary(scope_b, summary_id)
        if leaked is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


LONG_HORIZON_MEMORY_STORE_CHECKS: tuple[MemoryProviderQualificationCheck, ...] = (
    LongHorizonTenantIsolationCheck(),
)
