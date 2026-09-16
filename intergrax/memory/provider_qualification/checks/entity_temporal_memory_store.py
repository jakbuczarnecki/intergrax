# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationDirection,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    RelationTypeRef,
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.provider_qualification import (
    EntityTemporalMemoryStoreQualificationCheck,
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
)
from intergrax.memory.provider_qualification.checks._helpers import failed, passed
from intergrax.memory.provider_qualification.checks._suite import validate_canonical_check_suite

_CAPABILITY = MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED

_T0 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
_T2 = datetime(2025, 6, 3, 12, 0, 0, tzinfo=timezone.utc)
_T3 = datetime(2025, 6, 4, 12, 0, 0, tzinfo=timezone.utc)


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


def _memory_entity(
    scope: EntityMemoryScope,
    context: MemoryProviderQualificationContext,
    *,
    suffix: str,
    canonical_name: str,
    revision: int,
) -> EntityRecord:
    memory_id = f"{suffix}-{context.qualification_run_id}"
    entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
    return EntityRecord(
        entity_id=entity_id,
        entity_type=EntityTypeRef("qualification"),
        canonical_name=canonical_name,
        source_memory_id=memory_id,
        source_memory_revision=revision,
    )


def _endpoint_entity(entity_id: str, name: str) -> EntityRecord:
    return EntityRecord(
        entity_id=entity_id,
        entity_type=EntityTypeRef("qualification"),
        canonical_name=name,
    )


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
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        record = _memory_entity(
            scope_a,
            context,
            suffix="tenant-iso",
            canonical_name="tenant-a-only",
            revision=1,
        )
        store.upsert_entity(scope_a, record)
        leaked = store.get_entity(
            EntityMemoryScope(
                tenant_id=_tenant_b(context),
                user_id=context.user_qualification_id,
            ),
            record.entity_id,
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
class EntityTemporalUserIsolationCheck:
    check_id: str = "entity_temporal.user_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-a",
        )
        scope_b = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-b",
        )
        memory_id = f"user-iso-{context.qualification_run_id}"
        record_a = _memory_entity(
            scope_a,
            context,
            suffix="user-iso",
            canonical_name="user-a-marker",
            revision=1,
        )
        sibling_entity_id = entity_memory_entity_id_for_entry(scope_b, memory_id)
        store.upsert_entity(scope_a, record_a)
        if store.get_entity(scope_b, sibling_entity_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalWorkspaceIsolationCheck:
    check_id: str = "entity_temporal.workspace_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-a",
        )
        scope_b = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
            workspace_id=f"{context.workspace_qualification_id}-b",
        )
        memory_id = f"ws-iso-{context.qualification_run_id}"
        record_a = _memory_entity(
            scope_a,
            context,
            suffix="ws-iso",
            canonical_name="workspace-a-marker",
            revision=1,
        )
        sibling_entity_id = entity_memory_entity_id_for_entry(scope_b, memory_id)
        if sibling_entity_id == record_a.entity_id:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.WORKSPACE_ISOLATION_FAILURE,
                detail="projection_identity_collision",
            )
        store.upsert_entity(scope_a, record_a)
        if store.get_entity(scope_b, sibling_entity_id) is not None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.WORKSPACE_ISOLATION_FAILURE,
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
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=context.user_qualification_id,
        )
        first = _memory_entity(
            scope,
            context,
            suffix="stale-rev",
            canonical_name="revision-two",
            revision=2,
        )
        store.upsert_entity(scope, first)
        stale = _memory_entity(
            scope,
            context,
            suffix="stale-rev",
            canonical_name="stale-overwrite-attempt",
            revision=1,
        )
        store.upsert_entity(scope, stale)
        stored = store.get_entity(scope, first.entity_id)
        if stored is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        if stored.source_memory_revision != 2 or stored.canonical_name != "revision-two":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalSameRevisionIdempotencyCheck:
    check_id: str = "entity_temporal.same_revision_idempotency"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-idem",
        )
        record = _memory_entity(
            scope,
            context,
            suffix="same-rev-idem",
            canonical_name="stable-name",
            revision=3,
        )
        first = store.upsert_entity(scope, record)
        second = store.upsert_entity(scope, record)
        if first != second:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE,
            )
        stored = store.get_entity(scope, record.entity_id)
        if stored is None or stored.canonical_name != "stable-name":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalHigherRevisionCheck:
    check_id: str = "entity_temporal.higher_source_revision"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-higher",
        )
        initial = _memory_entity(
            scope,
            context,
            suffix="higher-rev",
            canonical_name="rev-one",
            revision=1,
        )
        store.upsert_entity(scope, initial)
        updated = _memory_entity(
            scope,
            context,
            suffix="higher-rev",
            canonical_name="rev-two",
            revision=2,
        )
        store.upsert_entity(scope, updated)
        stored = store.get_entity(scope, initial.entity_id)
        if stored is None:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        if stored.source_memory_revision != 2 or stored.canonical_name != "rev-two":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalSameRevisionUpdateCheck:
    check_id: str = "entity_temporal.same_revision_memory_sourced_update"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        """Memory-sourced projections accept same-revision payload replacement."""
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-same-upd",
        )
        first = _memory_entity(
            scope,
            context,
            suffix="same-upd",
            canonical_name="before",
            revision=5,
        )
        store.upsert_entity(scope, first)
        replacement = _memory_entity(
            scope,
            context,
            suffix="same-upd",
            canonical_name="after",
            revision=5,
        )
        store.upsert_entity(scope, replacement)
        stored = store.get_entity(scope, first.entity_id)
        if stored is None or stored.canonical_name != "after":
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
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope_a = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-a",
        )
        scope_b = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-b",
        )
        memory_a = f"del-a-{context.qualification_run_id}"
        memory_b = f"del-b-{context.qualification_run_id}"
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


@dataclass(frozen=True, slots=True)
class EntityTemporalDeleteIdempotencyCheck:
    check_id: str = "entity_temporal.delete_idempotency"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-del-idem",
        )
        memory_id = f"del-idem-{context.qualification_run_id}"
        record = _memory_entity(
            scope,
            context,
            suffix="del-idem",
            canonical_name="to-delete",
            revision=1,
        )
        store.upsert_entity(scope, record)
        first = store.delete_by_source_memory(scope, memory_id)
        second = store.delete_by_source_memory(scope, memory_id)
        missing = store.delete_by_source_memory(scope, "missing-memory-id")
        if first < 1 or second != 0 or missing != 0:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.INVALID_FAILURE_BEHAVIOR,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class EntityTemporalRelationTemporalSemanticsCheck:
    check_id: str = "entity_temporal.relation_temporal_semantics"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        scope = EntityMemoryScope(
            tenant_id=_tenant_a(context),
            user_id=f"{context.user_qualification_id}-temporal",
        )
        source_id = f"ent-src-{context.qualification_run_id}"
        target_id = f"ent-tgt-{context.qualification_run_id}"
        store.upsert_entity(scope, _endpoint_entity(source_id, "source"))
        store.upsert_entity(scope, _endpoint_entity(target_id, "target"))
        relation_id = f"rel-temporal-{context.qualification_run_id}"
        relation = EntityRelationRecord(
            relation_id=relation_id,
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=RelationTypeRef("qualification_link"),
            valid_from=_T1.isoformat(),
            valid_until=_T3.isoformat(),
        )
        store.upsert_relation(scope, relation)

        before = store.query_relations(
            scope,
            EntityRelationQuery(entity_id=source_id, as_of=_T0),
        )
        if before.relations:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TEMPORAL_SEMANTICS_FAILURE,
                detail="before_valid_from",
            )

        inside = store.query_relations(
            scope,
            EntityRelationQuery(entity_id=source_id, as_of=_T2),
        )
        if len(inside.relations) != 1:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TEMPORAL_SEMANTICS_FAILURE,
                detail="inside_window",
            )

        at_until = store.query_relations(
            scope,
            EntityRelationQuery(entity_id=source_id, as_of=_T3),
        )
        if at_until.relations:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TEMPORAL_SEMANTICS_FAILURE,
                detail="valid_until_exclusive",
            )

        open_relation_id = f"rel-open-{context.qualification_run_id}"
        store.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id=open_relation_id,
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=RelationTypeRef("qualification_open"),
                valid_from=_T1.isoformat(),
                valid_until=None,
            ),
        )
        open_future = store.query_relations(
            scope,
            EntityRelationQuery(entity_id=source_id, as_of=_T3),
        )
        open_ids = {item.relation_id for item in open_future.relations}
        if open_relation_id not in open_ids:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TEMPORAL_SEMANTICS_FAILURE,
                detail="open_ended",
            )

        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


ENTITY_TEMPORAL_MEMORY_STORE_CHECKS: tuple[
    EntityTemporalMemoryStoreQualificationCheck, ...
] = (
    EntityTemporalTenantIsolationCheck(),
    EntityTemporalUserIsolationCheck(),
    EntityTemporalWorkspaceIsolationCheck(),
    EntityTemporalStaleRevisionCheck(),
    EntityTemporalSameRevisionIdempotencyCheck(),
    EntityTemporalHigherRevisionCheck(),
    EntityTemporalSameRevisionUpdateCheck(),
    EntityTemporalDeleteScopeCheck(),
    EntityTemporalDeleteIdempotencyCheck(),
    EntityTemporalRelationTemporalSemanticsCheck(),
)

validate_canonical_check_suite(
    ENTITY_TEMPORAL_MEMORY_STORE_CHECKS,
    capability=_CAPABILITY,
    check_id_of=lambda item: item.check_id,
    capability_of=lambda item: item.capability,
    severity_of=lambda item: item.severity,
)


def default_entity_temporal_checks() -> tuple[EntityTemporalMemoryStoreQualificationCheck, ...]:
    return ENTITY_TEMPORAL_MEMORY_STORE_CHECKS
