# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13B behavioral qualification coverage."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.memory.contracts.entity_temporal_memory import EntityRelationResult
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationStatus,

)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.provider_qualification.checks.user_profile_store import (
    UserProfileTenantIsolationCheck,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from intergrax.memory.stores.in_memory_session_turn_index_store import (
    InMemorySessionTurnIndexStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from tests.unit.memory.test_mem_ent13_provider_qualification import (
    _StaticFactory,
    _StaleOverwriteEntityStore,
    _context,
)

pytestmark = pytest.mark.unit


class _TemporalBlindEntityStore(InMemoryEntityTemporalMemoryStore):
    def query_relations(self, scope, query):  # type: ignore[no-untyped-def]
        if not self._relations:
            return EntityRelationResult(relations=())
        return EntityRelationResult(relations=(next(iter(self._relations.values())),))


class _ProcedureStaleOverwriteStore(InMemoryProceduralMemoryStore):
    def upsert_procedure(self, scope, record):  # type: ignore[no-untyped-def]
        from intergrax.memory.stores.in_memory_procedural_memory_store import _storage_key

        storage_key = _storage_key(scope, record.procedure_id)
        self._records[storage_key] = record
        return record


class _LongHorizonSourceLossStore(InMemoryLongHorizonMemoryStore):
    def get_summary(self, scope, summary_id):  # type: ignore[no-untyped-def]
        from dataclasses import replace

        from intergrax.memory.contracts.long_horizon_memory import MemorySourceRef

        stored = super().get_summary(scope, summary_id)
        if stored is None or not stored.source_memory_refs:
            return stored
        corrupted = replace(stored.source_memory_refs[0], revision=stored.source_memory_refs[0].revision + 99)
        return replace(stored, source_memory_refs=(corrupted,))


class _SessionTurnTenantLeakStore(InMemorySessionTurnIndexStore):
    async def search_turns(self, **kwargs):  # type: ignore[no-untyped-def]
        return await super().search_turns(
            query=kwargs.get("query", ""),
            tenant_id="",
            session_id=kwargs.get("session_id"),
            user_id=kwargs.get("user_id"),
        )


@dataclass(frozen=True, slots=True)
class _WeakenedCanonicalOverride:
    check_id: str = UserProfileTenantIsolationCheck().check_id

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return MemoryProviderCapabilityKind.USER_PROFILE_STORE

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return MemoryProviderCheckSeverity.OPTIONAL

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        return MemoryProviderCheckResult(
            check_id=self.check_id,
            capability=self.capability,
            severity=self.severity,
            passed=True,
        )


class _EmptyChecksRunnerFactory(MemoryProviderInstanceFactory[UserProfileStore]):
    async def create(self) -> UserProfileStore:
        return InMemoryUserProfileStore()

    async def dispose(self, instance: UserProfileStore) -> None:
        return None


@pytest.mark.asyncio
async def test_required_zero_canonical_checks_not_qualified() -> None:
    runner = MemoryProviderQualificationRunner()

    class _NoSuiteRunner(MemoryProviderQualificationRunner):
        pass

    runner = _NoSuiteRunner()
    from intergrax.memory.provider_qualification.checks import user_profile_store as ups

    original = ups.USER_PROFILE_STORE_CHECKS
    ups.USER_PROFILE_STORE_CHECKS = ()
    try:
        result = await runner.qualify(
            descriptor=MemoryProviderDescriptor(
                provider_id="zero.checks",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("zero-checks"),
            request=MemoryProviderQualificationRequest(
                required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_EmptyChecksRunnerFactory(),
            ),
        )
    finally:
        ups.USER_PROFILE_STORE_CHECKS = original

    cap = result.capability_results[0]
    assert cap.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert cap.checks_executed == 0
    assert (
        MemoryProviderQualificationFailureReason.QUALIFICATION_COVERAGE_MISSING in cap.reason_codes
    )


@pytest.mark.asyncio
async def test_custom_check_cannot_override_canonical_id() -> None:
    runner = MemoryProviderQualificationRunner(
        extra_user_profile_checks=(_WeakenedCanonicalOverride(),),
    )
    with pytest.raises(ValueError, match="cannot override canonical"):
        await runner.qualify(
            descriptor=MemoryProviderDescriptor(
                provider_id="override.attempt",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("override"),
            request=MemoryProviderQualificationRequest(
                required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_StaticFactory(lambda: InMemoryUserProfileStore()),
            ),
        )


@pytest.mark.asyncio
async def test_temporal_blind_entity_store_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.temporal",
            capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        context=_context("temporal-fail"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            entity_temporal_memory_store=_StaticFactory(lambda: _TemporalBlindEntityStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.TEMPORAL_SEMANTICS_FAILURE in result.reason_codes
    )


@pytest.mark.asyncio
async def test_procedure_stale_overwrite_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.proc.stale",
            capabilities=(MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE,),
        ),
        context=_context("proc-stale"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            procedure_memory_store=_StaticFactory(lambda: _ProcedureStaleOverwriteStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE in result.reason_codes
    )


@pytest.mark.asyncio
async def test_long_horizon_source_fidelity_loss_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.lh.source",
            capabilities=(MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE,),
        ),
        context=_context("lh-source"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            long_horizon_memory_store=_StaticFactory(lambda: _LongHorizonSourceLossStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert MemoryProviderQualificationFailureReason.SOURCE_FIDELITY_FAILURE in result.reason_codes


@pytest.mark.asyncio
async def test_session_turn_index_reference_qualifies() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.session_turn_index",
            capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        context=_context("sti-ref"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            session_turn_index_store=_StaticFactory(lambda: InMemorySessionTurnIndexStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    cap = result.capability_results[0]
    assert cap.checks_executed >= 4
    assert cap.checks_failed == 0


@pytest.mark.asyncio
async def test_session_turn_tenant_leak_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.sti.tenant",
            capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        context=_context("sti-leak"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.SESSION_TURN_INDEX_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            session_turn_index_store=_StaticFactory(lambda: _SessionTurnTenantLeakStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED


@pytest.mark.asyncio
async def test_stale_revision_still_detected() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.stale_revision",
            capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        context=_context("stale-still"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            entity_temporal_memory_store=_StaticFactory(lambda: _StaleOverwriteEntityStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
