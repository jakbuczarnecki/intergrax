# © Artur Czarnecki. All rights reserved.

"""Typed E2E composition harness for Memory certification (MEM-ENT-15)."""

from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityTemporalMemoryStore,
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlPlane,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_observability import RecordingMemoryObservabilitySink
from intergrax.memory.default_memory_control_plane import DefaultMemoryControlPlane
from intergrax.applications._shared.entity_user_profile_memory_projection import (
    EntityIndexerUserProfileMemoryProjection,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryIndexer
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.entity_temporal_memory_service import EntityTemporalMemoryService
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.utils.time_provider import TimeProvider
from tests.unit.memory.resilience.recovery_projection import FailOnceRepairableProjection

_RUN_ID = "mem-ent15-e2e"


@dataclass(slots=True)
class RecordingEntityMemoryIndexer:
    """Records trusted ``RequestIdentity`` values passed into entity indexing."""

    inner: EntityMemoryIndexer
    observed_indexing: list[RequestIdentity] = field(default_factory=list)

    def index_memory_entry(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entry: UserProfileMemoryEntry,
    ) -> object:
        self.observed_indexing.append(identity)
        return self.inner.index_memory_entry(identity, scope, entry)


class FixedUtcTimeProvider(TimeProvider):
    """Deterministic wall clock for control-plane recall pipelines."""

    _frozen: datetime = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def set_frozen(cls, moment: datetime) -> None:
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        cls._frozen = moment

    @classmethod
    def utc_now(cls) -> datetime:
        return cls._frozen


@dataclass(slots=True)
class MemoryE2EHarness:
    tenant_id: str
    user_id: str
    store: UserProfileStore
    manager: UserProfileManager
    plane: MemoryControlPlane
    observability: RecordingMemoryObservabilitySink
    entity_store: EntityTemporalMemoryStore | None = None
    entity_projection: EntityIndexerUserProfileMemoryProjection | None = None
    entity_indexer: RecordingEntityMemoryIndexer | None = None
    sqlite_path: Path | None = None
    run_id: str = _RUN_ID

    def identity(self, *, user_id: str | None = None, tenant_id: str | None = None) -> RequestIdentity:
        uid = user_id if user_id is not None else self.user_id
        return RequestIdentity(
            tenant_id=tenant_id if tenant_id is not None else self.tenant_id,
            user_id=uid,
            principal_type=PrincipalType.USER,
            auth_subject=uid,
        )

    def user_scope(self, identity: RequestIdentity | None = None) -> object:
        return user_memory_scope(identity or self.identity())

    async def canonical_profile(self, user_id: str | None = None) -> UserProfile:
        uid = user_id if user_id is not None else self.user_id
        return await self.manager.get_profile(uid)

    async def close_store(self) -> None:
        if isinstance(self.store, SQLiteUserProfileStore):
            self.store.close()

    def entity_record_for_entry(self, user_id: str, entry_id: str) -> object | None:
        if self.entity_store is None:
            return None
        scope = EntityMemoryScope(tenant_id=self.tenant_id, user_id=user_id)
        entity_id = entity_memory_entity_id_for_entry(scope, entry_id)
        return self.entity_store.get_entity(scope, entity_id)


DisposeStore = Callable[[UserProfileStore], Awaitable[None]]


async def _noop_dispose(_store: UserProfileStore) -> None:
    return None


def _build_core(
    *,
    tenant_id: str,
    user_id: str,
    store: UserProfileStore,
    observability: RecordingMemoryObservabilitySink | None = None,
    governance: MemorySecurityGovernanceService | None = None,
    projections: Sequence[object] | None = None,
    time_provider: type[TimeProvider] = FixedUtcTimeProvider,
    include_entity_projection: bool = True,
) -> MemoryE2EHarness:
    recording = observability or RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    governance_service = governance or build_default_memory_security_governance_service(
        diagnostic_emitter=emitter,
    )
    entity_store: EntityTemporalMemoryStore | None = None
    entity_projection: EntityIndexerUserProfileMemoryProjection | None = None
    recording_indexer: RecordingEntityMemoryIndexer | None = None
    resolved_projections: list[object] = list(projections or ())
    if include_entity_projection:
        entity_store = InMemoryEntityTemporalMemoryStore()
        inner_indexer = DefaultEntityMemoryIndexer(
            entity_store,
            security_governance=governance_service,
            diagnostic_emitter=emitter,
        )
        recording_indexer = RecordingEntityMemoryIndexer(inner=inner_indexer)
        entity_capability = EntityTemporalMemoryService(
            _store=entity_store,
            _security_governance=governance_service,
        )
        entity_projection = EntityIndexerUserProfileMemoryProjection(
            indexer=recording_indexer,
            entity_capability=entity_capability,
        )
        resolved_projections.append(entity_projection)
    manager = UserProfileManager(
        store,
        tenant_id=tenant_id,
        memory_projections=tuple(resolved_projections),
        diagnostic_emitter=emitter,
    )
    plane = build_default_memory_control_plane(
        user_profile_manager=manager,
        security_governance=governance_service,
        memory_observability_sink=recording,
        memory_diagnostic_emitter=emitter,
    )
    if isinstance(plane, DefaultMemoryControlPlane):
        plane.time_provider = time_provider
    return MemoryE2EHarness(
        tenant_id=tenant_id,
        user_id=user_id,
        store=store,
        manager=manager,
        plane=plane,
        observability=recording,
        entity_store=entity_store,
        entity_projection=entity_projection,
        entity_indexer=recording_indexer,
    )


def build_in_memory_memory_harness(
    *,
    tenant_id: str = "tenant-ent15",
    user_id: str = "user-ent15",
    governance: MemorySecurityGovernanceService | None = None,
    extra_projections: Sequence[object] | None = None,
    include_entity_projection: bool = True,
) -> MemoryE2EHarness:
    return _build_core(
        tenant_id=tenant_id,
        user_id=user_id,
        store=InMemoryUserProfileStore(),
        governance=governance,
        projections=extra_projections,
        include_entity_projection=include_entity_projection,
    )


def build_in_memory_memory_harness_with_recovery_projection(
    *,
    tenant_id: str = "tenant-ent15-recovery",
    user_id: str = "user-ent15-recovery",
) -> tuple[MemoryE2EHarness, FailOnceRepairableProjection]:
    recovery = FailOnceRepairableProjection()
    harness = _build_core(
        tenant_id=tenant_id,
        user_id=user_id,
        store=InMemoryUserProfileStore(),
        projections=(recovery,),
        include_entity_projection=False,
    )
    return harness, recovery


def build_sqlite_memory_harness(
    *,
    db_path: Path,
    tenant_id: str = "tenant-ent15-sqlite",
    user_id: str = "user-ent15-sqlite",
) -> MemoryE2EHarness:
    harness = _build_core(
        tenant_id=tenant_id,
        user_id=user_id,
        store=SQLiteUserProfileStore(str(db_path)),
        include_entity_projection=True,
    )
    harness.sqlite_path = db_path
    return harness


def build_sqlite_memory_harness_from_path(
    db_path: Path,
    *,
    tenant_id: str,
    user_id: str,
) -> MemoryE2EHarness:
    return build_sqlite_memory_harness(
        db_path=db_path,
        tenant_id=tenant_id,
        user_id=user_id,
    )


def build_plugin_user_profile_harness(
    *,
    create_store: Callable[[], UserProfileStore],
    tenant_id: str = "tenant-ent15-plugin",
    user_id: str = "user-ent15-plugin",
) -> MemoryE2EHarness:
    return _build_core(
        tenant_id=tenant_id,
        user_id=user_id,
        store=create_store(),
        include_entity_projection=True,
    )


@dataclass(slots=True)
class SqliteMemoryCompositionSession:
    """Owns tempdir + path for durable recomposition proofs."""

    directory: tempfile.TemporaryDirectory[str]
    db_path: Path

    @classmethod
    def create(cls) -> SqliteMemoryCompositionSession:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "user_profiles.db"
        return cls(directory=directory, db_path=path)

    def cleanup(self) -> None:
        self.directory.cleanup()
