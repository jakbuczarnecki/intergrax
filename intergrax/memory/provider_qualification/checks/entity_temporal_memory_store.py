# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    entity_memory_entity_id_for_entry,
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

_CAPABILITY = MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


@dataclass(frozen=True, slots=True)
class EntityTemporalTenantIsolationCheck:
    check_id: str = "entity_temporal.tenant_isolation"

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
        if not isinstance(store, EntityTemporalMemoryStore):
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        memory_id = f"mem-{context.qualification_run_id}"
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        entity_id = entity_memory_entity_id_for_entry(scope_a, memory_id)
        record = EntityRecord(
            entity_id=entity_id,
            entity_type=EntityTypeRef("qualification"),
            canonical_name="tenant-a-only",
            source_memory_id=memory_id,
            source_memory_revision=1,
        )
        store.upsert_entity(scope_a, record)
        leaked = store.get_entity(
            EntityMemoryScope(
                tenant_id=_tenant_b(context),
                user_id=context.user_qualification_id,
            ),
            entity_id,
        )
        if leaked is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalStaleRevisionCheck:
    check_id: str = "entity_temporal.stale_source_revision"

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
        if not isinstance(store, EntityTemporalMemoryStore):
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        memory_id = f"rev-{context.qualification_run_id}"
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
        first = EntityRecord(
            entity_id=entity_id,
            entity_type=EntityTypeRef("qualification"),
            canonical_name="revision-two",
            source_memory_id=memory_id,
            source_memory_revision=2,
        )
        store.upsert_entity(scope, first)
        stale = EntityRecord(
            entity_id=entity_id,
            entity_type=EntityTypeRef("qualification"),
            canonical_name="stale-overwrite-attempt",
            source_memory_id=memory_id,
            source_memory_revision=1,
        )
        result = store.upsert_entity(scope, stale)
        if result.canonical_name != "revision-two":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalDeleteScopeCheck:
    check_id: str = "entity_temporal.delete_scope"

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
        if not isinstance(store, EntityTemporalMemoryStore):
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.CONTRACT_MISMATCH,
            )
        memory_a = f"del-a-{context.qualification_run_id}"
        memory_b = f"del-b-{context.qualification_run_id}"
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-a",
        )
        scope_b = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-b",
        )
        entity_a = entity_memory_entity_id_for_entry(scope_a, memory_a)
        entity_b = entity_memory_entity_id_for_entry(scope_b, memory_b)
        store.upsert_entity(
            scope_a,
            EntityRecord(
                entity_id=entity_a,
                entity_type=EntityTypeRef("qualification"),
                canonical_name="delete-a",
                source_memory_id=memory_a,
                source_memory_revision=1,
            ),
        )
        store.upsert_entity(
            scope_b,
            EntityRecord(
                entity_id=entity_b,
                entity_type=EntityTypeRef("qualification"),
                canonical_name="delete-b",
                source_memory_id=memory_b,
                source_memory_revision=1,
            ),
        )
        store.delete_by_source_memory(scope_a, memory_a)
        if store.get_entity(scope_b, entity_b) is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.DELETE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


ENTITY_TEMPORAL_MEMORY_STORE_CHECKS: tuple[MemoryProviderQualificationCheck, ...] = (
    EntityTemporalTenantIsolationCheck(),
    EntityTemporalStaleRevisionCheck(),
    EntityTemporalDeleteScopeCheck(),
)
