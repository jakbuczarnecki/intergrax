# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-007 Phase 4C — EffectiveRosterAuthorityService unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.errors import (
    EffectiveRosterAuthorityConflict,
    EffectiveRosterAuthorityNotFound,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("a" * 64)
_APP = "demo_app"
_ENV = "env-prod"
_RELEASE = "rel-1"


def _entry() -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id="search",
        installation_slot_id="slot-search",
        active_installation_id="inst-search-1",
        package_digest=_DIGEST,
        distribution_package_id="intergrax-local-search-agent",
        effective_enablement=True,
        merged_config={"limits": {"rpm": 100}},
        factory_reference=AgentBindingFactoryReference(
            builder_key="search-builder",
            factory_path="agents.search.factory:build",
        ),
    )


def _roster(
    *,
    application_id: str = _APP,
    application_environment_id: str = _ENV,
    manifest_release_id: str = _RELEASE,
) -> EffectiveRoster:
    return EffectiveRoster(
        application_id=application_id,
        application_environment_id=application_environment_id,
        manifest_release_id=manifest_release_id,
        entries=(_entry(),),
    ).with_revision_id()


def _revision(
    *,
    effective_roster_revision_id: str,
    application_id: str = _APP,
    application_environment_id: str = _ENV,
    application_release_id: str = _RELEASE,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id="rev-authority-1",
        application_id=application_id,
        application_environment_id=application_environment_id,
        application_release_id=application_release_id,
        platform_version="0.1.0",
        effective_roster_revision_id=effective_roster_revision_id,
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest="artifact-digest",
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )


@dataclass
class _FakeEffectiveRosterSnapshotStore:
    snapshots: dict[str, EffectiveRoster] = field(default_factory=dict)

    def get_by_revision(
        self,
        effective_roster_revision_id: str,
    ) -> EffectiveRoster | None:
        return self.snapshots.get(effective_roster_revision_id)

    def persist(self, roster: EffectiveRoster) -> EffectiveRoster:
        revision_id = roster.effective_roster_revision_id
        if revision_id is None:
            raise ValueError("roster lacks revision identity")
        self.snapshots[revision_id] = roster
        return roster


def _service(
    store: _FakeEffectiveRosterSnapshotStore | None = None,
) -> tuple[EffectiveRosterAuthorityService, _FakeEffectiveRosterSnapshotStore]:
    backing = store or _FakeEffectiveRosterSnapshotStore()
    return EffectiveRosterAuthorityService(snapshot_store=backing), backing


def test_valid_snapshot_returns_exact_effective_roster() -> None:
    service, store = _service()
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    store.snapshots[revision_id] = roster
    revision = _revision(effective_roster_revision_id=revision_id)
    resolved = service.require_for_revision(revision)
    assert resolved is roster
    assert resolved == roster


def test_missing_snapshot_raises_authority_not_found() -> None:
    service, _store = _service()
    revision = _revision(effective_roster_revision_id="sha256:" + ("f" * 64))
    with pytest.raises(EffectiveRosterAuthorityNotFound):
        service.require_for_revision(revision)


def test_embedded_revision_id_mismatch_raises_conflict() -> None:
    service, store = _service()
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    corrupt = roster.model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("b" * 64)}
    )
    store.snapshots[revision_id] = corrupt
    revision = _revision(effective_roster_revision_id=revision_id)
    with pytest.raises(
        EffectiveRosterAuthorityConflict,
        match="revision id mismatch",
    ):
        service.require_for_revision(revision)


def test_recomputed_hash_mismatch_raises_conflict() -> None:
    service, store = _service()
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    corrupt = roster.model_copy(update={"manifest_release_id": "rel-other"}).model_copy(
        update={"effective_roster_revision_id": revision_id}
    )
    store.snapshots[revision_id] = corrupt
    revision = _revision(effective_roster_revision_id=revision_id)
    with pytest.raises(
        EffectiveRosterAuthorityConflict,
        match="content identity mismatch",
    ):
        service.require_for_revision(revision)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("application_id", "other-app"),
        ("application_environment_id", "other-env"),
        ("manifest_release_id", "other-release"),
    ],
)
def test_scope_or_release_mismatch_raises_conflict(
    field_name: str,
    field_value: str,
) -> None:
    service, store = _service()
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    alternate = roster.model_copy(update={field_name: field_value}).with_revision_id()
    alternate_id = alternate.effective_roster_revision_id
    assert alternate_id is not None
    store.snapshots[alternate_id] = alternate
    revision = _revision(effective_roster_revision_id=alternate_id)
    with pytest.raises(EffectiveRosterAuthorityConflict):
        service.require_for_revision(revision)


def test_service_uses_protocol_compatible_fake_store() -> None:
    service, store = _service()
    roster = _roster()
    revision_id = roster.effective_roster_revision_id
    assert revision_id is not None
    store.snapshots[revision_id] = roster
    revision = _revision(effective_roster_revision_id=revision_id)
    assert service.require_for_revision(revision) == roster


def test_service_has_no_desired_state_dependencies() -> None:
    service = EffectiveRosterAuthorityService(
        snapshot_store=_FakeEffectiveRosterSnapshotStore()
    )
    assert not hasattr(service, "_roster_builder")
    assert not hasattr(service, "_installation_store")
    assert not hasattr(service, "_binding_store")


def test_admin_can_compose_with_custom_snapshot_store_and_authority() -> None:
    from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
    from intergrax.agent_distribution.activation import ActivationService
    from intergrax.agent_distribution.binding_service import BindingService
    from intergrax.agent_distribution.deployment import (
        FakeInMemoryRuntimeDeploymentAdapter,
    )
    from intergrax.agent_distribution.effective_roster import (
        EffectiveRosterBuilder,
        InstalledAgentRequirementSetBuilder,
    )
    from intergrax.agent_distribution.in_memory_stores import (
        AgentDistributionStoreState,
        InMemoryAgentArtifactMetadataStore,
        InMemoryAgentInstallationStore,
        InMemoryApplicationAgentBindingStore,
        InMemoryApplicationEnvironmentActivationStore,
        InMemoryApplicationEnvironmentServingStore,
        InMemoryDeploymentInstanceStore,
        InMemoryMaterializedRuntimeLockStore,
        InMemoryRuntimeMaterializationStore,
        InMemoryRuntimeRevisionStore,
    )
    from intergrax.agent_distribution.installation_service import InstallationService
    from intergrax.agent_distribution.runtime_revision_service import (
        RuntimeRevisionService,
    )

    store = _FakeEffectiveRosterSnapshotStore()
    authority = EffectiveRosterAuthorityService(snapshot_store=store)
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    service = AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=InMemoryApplicationAgentBindingStore(state),
        revision_store=InMemoryRuntimeRevisionStore(state),
        serving_store=InMemoryApplicationEnvironmentServingStore(state),
        deployment_instance_store=InMemoryDeploymentInstanceStore(state),
        lock_store=InMemoryMaterializedRuntimeLockStore(state),
        materialization_store=InMemoryRuntimeMaterializationStore(state),
        effective_roster_snapshot_store=store,
        effective_roster_authority=authority,
        artifact_metadata_store=InMemoryAgentArtifactMetadataStore(state),
        installation_service=InstallationService(installation_store),
        binding_service=BindingService(
            InMemoryApplicationAgentBindingStore(state),
            InstallationService(installation_store),
        ),
        revision_service=RuntimeRevisionService(InMemoryRuntimeRevisionStore(state)),
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(
            InMemoryAgentArtifactMetadataStore(state)
        ),
        activation_service=ActivationService(
            revision_store=InMemoryRuntimeRevisionStore(state),
            deployment_instance_store=InMemoryDeploymentInstanceStore(state),
            serving_store=InMemoryApplicationEnvironmentServingStore(state),
            activation_store=InMemoryApplicationEnvironmentActivationStore(state),
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
            projection_coordinator=object(),  # type: ignore[arg-type]
        ),
    )
    assert service._effective_roster_authority is authority
