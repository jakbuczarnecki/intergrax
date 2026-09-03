# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-007 Phase 4D — registry projection authority resolver tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import inspect
import pytest

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.errors import (
    EffectiveRosterAuthorityConflict,
    EffectiveRosterAuthorityNotFound,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection_authority_resolver import (
    RegistryProjectionAuthorityConflict,
    RegistryProjectionAuthorityNotFound,
    RegistryProjectionAuthorityResolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_DIGEST = "sha256:" + ("a" * 64)
_GRAPH_DIGEST = "sha256:" + ("d" * 64)
_ARTIFACT_DIGEST = "sha256:" + ("e" * 64)
_FACTORY_REF = AgentBindingFactoryReference(
    factory_path="example_agent.factory.build_agent",
)
_LOCK = MaterializedRuntimeLock(
    resolver_algorithm_id="intergrax.test",
    resolver_algorithm_version="1",
    inputs_digest="inputs-1",
    intergrax_version="0.1.0",
    python_version="3.12",
    packages=(),
    agent_closure=(),
).with_content_identity()
_LOCK_ID = _LOCK.lock_id or ""
_LOCK_DIGEST = _LOCK.lock_digest or ""


def _lock() -> MaterializedRuntimeLock:
    return _LOCK


def _entry(*, logical_agent_id: str = "search") -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id=logical_agent_id,
        installation_slot_id="slot-search",
        package_digest=_DIGEST,
        distribution_package_id="pkg-search",
        effective_enablement=True,
        factory_reference=_FACTORY_REF,
        manifest_origin_ref="manifest:agents/search",
    )


def _roster(*, logical_agent_id: str = "search") -> EffectiveRoster:
    return EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=_RELEASE,
        entries=(_entry(logical_agent_id=logical_agent_id),),
    ).with_revision_id()


def _revision(
    revision_id: str,
    *,
    roster: EffectiveRoster,
    lock_id: str = _LOCK_ID,
    lock_digest: str = _LOCK_DIGEST,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=roster.effective_roster_revision_id or "",
        installed_agent_package_digests=(_DIGEST,),
        materialized_runtime_lock_id=lock_id,
        materialized_runtime_lock_digest=lock_digest,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=_ARTIFACT_DIGEST,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
        activated_at=None,
    )


def _materialization(revision: RuntimeRevision) -> RuntimeMaterializationRecord:
    return RuntimeMaterializationRecord(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        artifact_locator="test:///artifact",
        materialization_artifact_digest=_ARTIFACT_DIGEST,
        materialized_runtime_lock_id=_LOCK_ID,
        materialized_runtime_lock_digest=_LOCK_DIGEST,
    )


@dataclass
class _FakeRevisionStore:
    revisions: dict[str, RuntimeRevision] = field(default_factory=dict)

    def get_revision(self, runtime_revision_id: str) -> RuntimeRevision | None:
        return self.revisions.get(runtime_revision_id)


@dataclass
class _FakeLockStore:
    locks: dict[str, MaterializedRuntimeLock] = field(default_factory=dict)

    def get_lock(self, lock_id: str) -> MaterializedRuntimeLock | None:
        return self.locks.get(lock_id)


@dataclass
class _FakeMaterializationStore:
    records: dict[str, RuntimeMaterializationRecord] = field(default_factory=dict)

    def get_by_revision(
        self,
        runtime_revision_id: str,
    ) -> RuntimeMaterializationRecord | None:
        return self.records.get(runtime_revision_id)


@dataclass
class _FakeSnapshotStore:
    snapshots: dict[str, EffectiveRoster] = field(default_factory=dict)

    def get_by_revision(
        self,
        effective_roster_revision_id: str,
    ) -> EffectiveRoster | None:
        return self.snapshots.get(effective_roster_revision_id)

    def persist(self, roster: EffectiveRoster) -> EffectiveRoster:
        revision_id = roster.effective_roster_revision_id
        assert revision_id is not None
        self.snapshots[revision_id] = roster
        return roster


def _resolver(
    *,
    revision_store: _FakeRevisionStore | None = None,
    snapshot_store: _FakeSnapshotStore | None = None,
    lock_store: _FakeLockStore | None = None,
    materialization_store: _FakeMaterializationStore | None = None,
) -> tuple[
    RegistryProjectionAuthorityResolver,
    _FakeRevisionStore,
    _FakeSnapshotStore,
    _FakeLockStore,
    _FakeMaterializationStore,
]:
    revisions = revision_store or _FakeRevisionStore()
    snapshots = snapshot_store or _FakeSnapshotStore()
    locks = lock_store or _FakeLockStore()
    materializations = materialization_store or _FakeMaterializationStore()
    return (
        RegistryProjectionAuthorityResolver(
            revision_store=revisions,
            effective_roster_authority=EffectiveRosterAuthorityService(
                snapshot_store=snapshots,
            ),
            lock_store=locks,
            materialization_store=materializations,
        ),
        revisions,
        snapshots,
        locks,
        materializations,
    )


def _seed_full_authority(
    *,
    revision_id: str = "rev-a",
    roster: EffectiveRoster | None = None,
) -> tuple[RegistryProjectionAuthorityResolver, RuntimeRevision, EffectiveRoster]:
    roster = roster or _roster()
    revision = _revision(revision_id, roster=roster)
    lock = _lock()
    resolver, revisions, snapshots, locks, materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    revisions.revisions[revision_id] = revision
    locks.locks[_LOCK_ID] = lock
    materializations.records[revision_id] = _materialization(revision)
    return resolver, revision, roster


def test_public_projection_api_has_no_effective_roster_param() -> None:
    signature = inspect.signature(
        build_production_registry_projection_input_bundle_for_revision
    )
    assert "effective_roster" not in signature.parameters


def test_resolver_returns_canonical_historical_roster() -> None:
    resolver, revision, roster = _seed_full_authority()
    resolved = resolver.require_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=revision.runtime_revision_id,
    )
    assert resolved.effective_roster is roster
    assert resolved.runtime_revision is revision


def test_resolver_uses_effective_roster_authority_service() -> None:
    resolver, revision, roster = _seed_full_authority()
    mutated = roster.model_copy(
        update={
            "entries": (
                roster.entries[0].model_copy(
                    update={"logical_agent_id": "mutated"},
                ),
            ),
        },
    ).with_revision_id()
    _, _, snapshots, _, _ = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    snapshots.snapshots[mutated.effective_roster_revision_id or ""] = mutated
    resolver, revisions, snapshots, locks, materializations = _resolver(
        snapshot_store=snapshots,
    )
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    materializations.records[revision.runtime_revision_id] = _materialization(revision)
    resolved = resolver.require_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=revision.runtime_revision_id,
    )
    assert resolved.effective_roster.entries[0].logical_agent_id == "search"


def test_missing_historical_snapshot_fails_closed() -> None:
    roster = _roster()
    revision = _revision("rev-missing-roster", roster=roster)
    resolver, revisions, _, locks, materializations = _resolver()
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    materializations.records[revision.runtime_revision_id] = _materialization(revision)
    with pytest.raises(
        RegistryProjectionAuthorityNotFound,
        match="canonical effective roster snapshot authority",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_invalid_historical_snapshot_scope_fails_closed() -> None:
    roster = _roster()
    revision = _revision("rev-bad-roster", roster=roster)
    bad_roster = roster.model_copy(update={"application_id": "app-b"})
    resolver, revisions, snapshots, locks, materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = bad_roster
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    materializations.records[revision.runtime_revision_id] = _materialization(revision)
    with pytest.raises(
        RegistryProjectionAuthorityConflict,
        match="content identity mismatch",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_missing_lock_fails_closed() -> None:
    roster = _roster()
    revision = _revision("rev-no-lock", roster=roster)
    resolver, revisions, snapshots, locks, materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    revisions.revisions[revision.runtime_revision_id] = revision
    materializations.records[revision.runtime_revision_id] = _materialization(revision)
    with pytest.raises(
        RegistryProjectionAuthorityNotFound,
        match="canonical materialized runtime lock not found",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_lock_digest_conflict_fails_closed() -> None:
    roster = _roster()
    revision = _revision(
        "rev-lock-conflict",
        roster=roster,
        lock_digest="sha256:" + ("9" * 64),
    )
    resolver, revisions, snapshots, locks, materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    materializations.records[revision.runtime_revision_id] = _materialization(
        revision
    ).model_copy(update={"materialized_runtime_lock_digest": "sha256:" + ("9" * 64)})
    with pytest.raises(
        RegistryProjectionAuthorityConflict,
        match="lock digest mismatch",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_missing_materialization_fails_closed() -> None:
    roster = _roster()
    revision = _revision("rev-no-materialization", roster=roster)
    resolver, revisions, snapshots, locks, _materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    with pytest.raises(
        RegistryProjectionAuthorityNotFound,
        match="missing canonical materialization record",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_materialization_conflict_fails_closed() -> None:
    roster = _roster()
    revision = _revision("rev-materialization-conflict", roster=roster)
    resolver, revisions, snapshots, locks, materializations = _resolver()
    snapshots.snapshots[roster.effective_roster_revision_id or ""] = roster
    revisions.revisions[revision.runtime_revision_id] = revision
    locks.locks[_LOCK_ID] = _lock()
    materializations.records[revision.runtime_revision_id] = _materialization(
        revision
    ).model_copy(update={"runtime_revision_id": "rev-other"})
    with pytest.raises(
        RegistryProjectionAuthorityConflict,
        match="materialization revision id mismatch",
    ):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )


def test_authority_service_errors_wrap_cleanly() -> None:
    roster = _roster()
    revision = _revision("rev-wrap", roster=roster)

    class _RaisingAuthority:
        def require_for_revision(self, _revision: RuntimeRevision) -> EffectiveRoster:
            raise EffectiveRosterAuthorityNotFound("missing roster authority")

    resolver = RegistryProjectionAuthorityResolver(
        revision_store=_FakeRevisionStore(
            revisions={revision.runtime_revision_id: revision},
        ),
        effective_roster_authority=_RaisingAuthority(),  # type: ignore[arg-type]
        lock_store=_FakeLockStore(locks={_LOCK_ID: _lock()}),
        materialization_store=_FakeMaterializationStore(
            records={revision.runtime_revision_id: _materialization(revision)},
        ),
    )
    with pytest.raises(RegistryProjectionAuthorityNotFound, match="missing roster"):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )

    class _ConflictAuthority:
        def require_for_revision(self, _revision: RuntimeRevision) -> EffectiveRoster:
            raise EffectiveRosterAuthorityConflict("roster hash mismatch")

    resolver = RegistryProjectionAuthorityResolver(
        revision_store=_FakeRevisionStore(
            revisions={revision.runtime_revision_id: revision},
        ),
        effective_roster_authority=_ConflictAuthority(),  # type: ignore[arg-type]
        lock_store=_FakeLockStore(locks={_LOCK_ID: _lock()}),
        materialization_store=_FakeMaterializationStore(
            records={revision.runtime_revision_id: _materialization(revision)},
        ),
    )
    with pytest.raises(RegistryProjectionAuthorityConflict, match="roster hash"):
        resolver.require_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
        )
