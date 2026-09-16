# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13 provider qualification framework."""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    RecordingMemoryObservabilitySink,
)
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
from intergrax.memory.memory_diagnostic_emitter import (
    MemoryDiagnosticEmitter,
    default_memory_diagnostic_emitter,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
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
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_store import UserProfileStore

pytestmark = pytest.mark.unit

T = TypeVar("T")


@dataclass(slots=True)
class _StaticFactory(MemoryProviderInstanceFactory[T]):
    _supplier: Callable[[], T]
    _disposer: Callable[[T], Awaitable[None]] | None = None

    async def create(self) -> T:
        return self._supplier()

    async def dispose(self, instance: T) -> None:
        if self._disposer is not None:
            await self._disposer(instance)


def _context(run_id: str = "run-qual-1") -> MemoryProviderQualificationContext:
    return MemoryProviderQualificationContext(
        qualification_run_id=run_id,
        tenant_qualification_id="qual-tenant",
        user_qualification_id="qual-user",
        workspace_qualification_id="qual-workspace",
        reference_time_iso="2025-01-01T00:00:00+00:00",
    )


def _in_memory_user_profile_factory() -> _StaticFactory[UserProfileStore]:
    return _StaticFactory(lambda: InMemoryUserProfileStore())


class _TenantLeakingUserProfileStore(InMemoryUserProfileStore):
    """Faulty reference: ignores tenant in storage key."""

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        key = ("", profile.identity.user_id)
        self._profiles[key] = profile

    async def get_profile(self, *, tenant_id: str, user_id: str) -> UserProfile:
        key = ("", user_id)
        if key in self._profiles:
            return self._profiles[key]
        return await super().get_profile(tenant_id=tenant_id, user_id=user_id)


class _StaleOverwriteEntityStore(InMemoryEntityTemporalMemoryStore):
    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        key = (scope.tenant_id, record.entity_id)
        self._entities[key] = record
        return record


class _DeleteLeakEntityStore(InMemoryEntityTemporalMemoryStore):
    def delete_by_source_memory(self, scope: EntityMemoryScope, source_memory_id: str) -> int:
        removed = 0
        for key in list(self._entities):
            if key[0] == scope.tenant_id:
                del self._entities[key]
                removed += 1
        for key in list(self._relations):
            if key[0] == scope.tenant_id:
                del self._relations[key]
                removed += 1
        return removed


class _DuplicatingSaveUserProfileStore(InMemoryUserProfileStore):
    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        await super().save_profile(tenant_id=tenant_id, profile=profile)
        bumped = UserProfile(
            identity=profile.identity,
            preferences=profile.preferences,
            system_instructions=profile.system_instructions,
            version=profile.version + 1,
        )
        await super().save_profile(tenant_id=tenant_id, profile=bumped)


class _BrokenCreateFactory(MemoryProviderInstanceFactory[UserProfileStore]):
    async def create(self) -> UserProfileStore:
        raise RuntimeError("plugin materialization failed")

    async def dispose(self, instance: UserProfileStore) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _CustomMarkerCheck:
    check_id: str = "custom.marker"

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


class _BrokenObservabilitySink(RecordingMemoryObservabilitySink):
    def record(self, event: object) -> None:
        raise RuntimeError("sink broken")


@pytest.mark.asyncio
async def test_reference_in_memory_user_profile_qualifies() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    cap = result.capability_results[0]
    assert cap.checks_failed == 0
    assert cap.checks_executed >= 6


@pytest.mark.asyncio
async def test_tenant_leak_fails_qualification() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.tenant_leak",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("run-tenant-leak"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(lambda: _TenantLeakingUserProfileStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE in result.reason_codes
    )


@pytest.mark.asyncio
async def test_stale_revision_overwrite_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.stale_revision",
            capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        context=_context("run-stale-rev"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            entity_temporal_memory_store=_StaticFactory(lambda: _StaleOverwriteEntityStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.REVISION_SEMANTICS_FAILURE in result.reason_codes
    )


@pytest.mark.asyncio
async def test_delete_leak_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.delete_leak",
            capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        context=_context("run-delete-leak"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            entity_temporal_memory_store=_StaticFactory(lambda: _DeleteLeakEntityStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.DELETE_ISOLATION_FAILURE in result.reason_codes
    )


@pytest.mark.asyncio
async def test_idempotency_violation_fails() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.idempotency",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("run-idem"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(lambda: _DuplicatingSaveUserProfileStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE in result.reason_codes


@pytest.mark.asyncio
async def test_optional_capability_unsupported_without_crash() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="partial.provider",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            optional_capabilities=(MemoryProviderCapabilityKind.SESSION_STORAGE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    session_cap = next(
        item
        for item in result.capability_results
        if item.capability is MemoryProviderCapabilityKind.SESSION_STORAGE
    )
    assert session_cap.status is MemoryProviderQualificationStatus.NOT_SUPPORTED


@pytest.mark.asyncio
async def test_required_capability_missing_factory_not_qualified() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="missing.session",
            capabilities=(),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.SESSION_STORAGE,),
        ),
        factories=MemoryProviderCapabilityFactories(),
    )
    assert result.status is MemoryProviderQualificationStatus.NOT_QUALIFIED


@pytest.mark.asyncio
async def test_materialization_failure_blocked() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="broken.materialization",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_BrokenCreateFactory(),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.BLOCKED


@pytest.mark.asyncio
async def test_dispose_called_after_failure() -> None:
    disposed = False

    async def _dispose(_instance: UserProfileStore) -> None:
        nonlocal disposed
        disposed = True

    runner = MemoryProviderQualificationRunner()
    await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="faulty.tenant_leak",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("run-cleanup"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(
                lambda: _TenantLeakingUserProfileStore(),
                _dispose,
            ),
        ),
    )
    assert disposed is True


@pytest.mark.asyncio
async def test_result_contains_no_secrets() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    serialized = repr(result)
    for token in ("password", "api_key", "connection string", "postgresql://"):
        assert token not in serialized.lower()


@pytest.mark.asyncio
async def test_deterministic_check_ordering() -> None:
    runner = MemoryProviderQualificationRunner()
    first = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("deterministic-a"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    second = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("deterministic-b"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    ids_first = [item.check_id for item in first.capability_results[0].check_results]
    ids_second = [item.check_id for item in second.capability_results[0].check_results]
    assert ids_first == ids_second
    assert ids_first == sorted(ids_first)


@pytest.mark.asyncio
async def test_custom_check_plugin() -> None:
    runner = MemoryProviderQualificationRunner(extra_user_profile_checks=(_CustomMarkerCheck(),))
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    check_ids = {item.check_id for item in result.capability_results[0].check_results}
    assert "custom.marker" in check_ids


@pytest.mark.asyncio
async def test_observer_failure_does_not_change_qualification() -> None:
    sink = _BrokenObservabilitySink()
    emitter = default_memory_diagnostic_emitter(sink)
    runner = MemoryProviderQualificationRunner(observability=emitter)
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED


@pytest.mark.asyncio
async def test_qualification_emits_terminal_provider_event() -> None:
    sink = RecordingMemoryObservabilitySink()
    emitter = default_memory_diagnostic_emitter(sink)
    runner = MemoryProviderQualificationRunner(observability=emitter)
    await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
        ),
    )
    terminal = [
        event
        for event in sink.events
        if event.operation is MemoryDiagnosticOperation.PROVIDER_QUALIFICATION
    ]
    assert terminal
    assert terminal[-1].outcome is MemoryDiagnosticOutcome.SUCCESS
    assert terminal[-1].provider_id == "in_memory.user_profile"


@pytest.mark.asyncio
async def test_sqlite_user_profile_qualifies() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "qual.db")
        holder: dict[str, SQLiteUserProfileStore] = {}

        def _create() -> SQLiteUserProfileStore:
            store = SQLiteUserProfileStore(db_path)
            holder["store"] = store
            return store

    async def _dispose(store: SQLiteUserProfileStore) -> None:
        store.close()

        runner = MemoryProviderQualificationRunner()
        result = await runner.qualify(
            descriptor=MemoryProviderDescriptor(
                provider_id="sqlite.user_profile",
                capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            context=_context("sqlite-run"),
            request=MemoryProviderQualificationRequest(
                required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            ),
            factories=MemoryProviderCapabilityFactories(
                user_profile_store=_StaticFactory(_create, _dispose),
            ),
        )
        assert result.status is MemoryProviderQualificationStatus.QUALIFIED


@pytest.mark.asyncio
async def test_multi_capability_partial_matrix() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.bundle",
            capabilities=(
                MemoryProviderCapabilityKind.USER_PROFILE_STORE,
                MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,
            ),
        ),
        context=_context("multi-cap"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            optional_capabilities=(MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_in_memory_user_profile_factory(),
            entity_temporal_memory_store=_StaticFactory(lambda: InMemoryEntityTemporalMemoryStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    statuses = {item.capability: item.status for item in result.capability_results}
    assert statuses[MemoryProviderCapabilityKind.USER_PROFILE_STORE] is (
        MemoryProviderQualificationStatus.QUALIFIED
    )
    assert statuses[MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE] is (
        MemoryProviderQualificationStatus.QUALIFIED
    )


@pytest.mark.asyncio
async def test_reference_entity_procedural_long_horizon() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="in_memory.full_stack",
            capabilities=(
                MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,
                MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE,
                MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE,
            ),
        ),
        context=_context("full-stack"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(
                MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE,
                MemoryProviderCapabilityKind.PROCEDURE_MEMORY_STORE,
                MemoryProviderCapabilityKind.LONG_HORIZON_MEMORY_STORE,
            ),
        ),
        factories=MemoryProviderCapabilityFactories(
            entity_temporal_memory_store=_StaticFactory(lambda: InMemoryEntityTemporalMemoryStore()),
            procedure_memory_store=_StaticFactory(lambda: InMemoryProceduralMemoryStore()),
            long_horizon_memory_store=_StaticFactory(lambda: InMemoryLongHorizonMemoryStore()),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED


@pytest.mark.asyncio
async def test_external_plugin_style_factory_without_runner_changes() -> None:
    """Third-party provider: only public factory + descriptor wiring."""

    class ExternalUserProfilePlugin:
        plugin_id = "external.qual.plugin"

        @staticmethod
        def build_store() -> UserProfileStore:
            return InMemoryUserProfileStore()

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id=ExternalUserProfilePlugin.plugin_id,
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            provider_version=None,
        ),
        context=_context("external"),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(ExternalUserProfilePlugin.build_store),
        ),
    )
    assert result.status is MemoryProviderQualificationStatus.QUALIFIED
    assert result.descriptor.provider_id == "external.qual.plugin"
