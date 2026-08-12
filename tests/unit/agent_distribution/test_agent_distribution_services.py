# © Artur Czarnecki. All rights reserved.

"""AP-4 installation and binding domain service tests."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.errors import (
    BindingLifecycleError,
    BindingRevisionConflict,
    InstallationLifecycleError,
    InstallationSlotConflict,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
)
from intergrax.agent_distribution.installation import InstallationState
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationStatus,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_PACKAGE_A = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
)
_PACKAGE_B = _PACKAGE_A.model_copy(
    update={"package_version": "2.0.0", "package_digest": _DIGEST_B}
)


def _trust_record() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
    )


def _services() -> tuple[InstallationService, BindingService, AgentDistributionStoreState]:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    return installation_service, binding_service, state


def _verified_installation(
    installation_service: InstallationService,
    *,
    installation_id: str,
    package_identity: AgentPackageIdentity = _PACKAGE_A,
) -> None:
    installation_service.create_candidate_installation(
        installation_id=installation_id,
        installation_slot_id="slot-search-prod",
        environment_id="env-prod",
        package_identity=package_identity,
    )
    installation_service.mark_verified(
        installation_id,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=_trust_record(),
    )


def test_installation_candidate_verified_active_flow() -> None:
    installation_service, _, _ = _services()
    installation_service.create_candidate_installation(
        installation_id="inst-v1",
        installation_slot_id="slot-search-prod",
        environment_id="env-prod",
        package_identity=_PACKAGE_A,
    )
    verified = installation_service.mark_verified(
        "inst-v1",
        artifact_store_ref="store://artifacts/v1",
        trust_record=_trust_record(),
    )
    assert verified.value.installation_state is InstallationState.VERIFIED
    active = installation_service.promote_verified_to_active("inst-v1")
    assert active.value.installation_state is InstallationState.INSTALLED_ACTIVE
    assert active.value.active_for_slot is True
    assert any(event.event_type == "installation.activated" for event in active.events)


def test_installation_invalid_transition_rejected() -> None:
    installation_service, _, _ = _services()
    installation_service.create_candidate_installation(
        installation_id="inst-v1",
        installation_slot_id="slot-search-prod",
        environment_id="env-prod",
        package_identity=_PACKAGE_A,
    )
    with pytest.raises(InstallationLifecycleError):
        installation_service.promote_verified_to_active("inst-v1")


def test_installation_exactly_one_active_per_slot() -> None:
    installation_service, _, state = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    _verified_installation(installation_service, installation_id="inst-v2", package_identity=_PACKAGE_B)
    installation_service.promote_verified_to_active("inst-v1")
    installation_service.promote_verified_to_active("inst-v2", expected_active_installation_id="inst-v1")
    active_ids = [
        record.installation_id
        for record in state.installations.values()
        if record.active_for_slot
    ]
    assert active_ids == ["inst-v2"]
    prior = state.installations["inst-v1"]
    assert prior.installation_state is InstallationState.INSTALLED_PREVIOUS
    assert prior.active_for_slot is False


def test_installation_upgrade_marks_previous_and_rollback_restores_digest() -> None:
    installation_service, _, state = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    _verified_installation(installation_service, installation_id="inst-v2", package_identity=_PACKAGE_B)
    installation_service.promote_verified_to_active("inst-v1")
    installation_service.promote_verified_to_active("inst-v2", expected_active_installation_id="inst-v1")
    rolled = installation_service.rollback_slot_to_previous(
        "slot-search-prod",
        expected_active_installation_id="inst-v2",
    )
    assert rolled.value.installation_id == "inst-v1"
    assert rolled.value.package_identity.package_digest == _DIGEST_A
    assert state.installations["inst-v2"].installation_state is InstallationState.INSTALLED_PREVIOUS


def test_installation_stale_expected_active_id_conflict() -> None:
    installation_service, _, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    _verified_installation(installation_service, installation_id="inst-v2", package_identity=_PACKAGE_B)
    installation_service.promote_verified_to_active("inst-v1")
    with pytest.raises(InstallationSlotConflict):
        installation_service.promote_verified_to_active("inst-v2", expected_active_installation_id="missing")


def test_installation_revoke_and_tombstone() -> None:
    installation_service, _, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    _verified_installation(installation_service, installation_id="inst-v2", package_identity=_PACKAGE_B)
    installation_service.promote_verified_to_active("inst-v1")
    installation_service.promote_verified_to_active("inst-v2", expected_active_installation_id="inst-v1")
    installation_service.revoke_installation("inst-v1")
    tombstoned = installation_service.tombstone_installation("inst-v1")
    assert tombstoned.value.installation_state is InstallationState.REMOVED_TOMBSTONE


def test_binding_create_update_enable_disable() -> None:
    installation_service, binding_service, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    installation_service.promote_verified_to_active("inst-v1")
    binding_service.create_binding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        config={"mode": "fast"},
    )
    updated = binding_service.update_config(
        "bind-1",
        {"mode": "accurate"},
        expected_revision=0,
    )
    assert updated.value.binding_revision == 1
    assert dict(updated.value.config) == {"mode": "accurate"}
    enabled = binding_service.enable("bind-1", expected_revision=1)
    assert enabled.value.enablement is True
    disabled = binding_service.disable("bind-1", expected_revision=2)
    assert disabled.value.enablement is False


def test_binding_stale_revision_conflict() -> None:
    installation_service, binding_service, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    installation_service.promote_verified_to_active("inst-v1")
    binding_service.create_binding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
    )
    with pytest.raises(BindingRevisionConflict):
        binding_service.update_config("bind-1", {"mode": "x"}, expected_revision=5)


def test_binding_tombstone_cannot_enable() -> None:
    installation_service, binding_service, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    installation_service.promote_verified_to_active("inst-v1")
    binding_service.create_binding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
    )
    binding_service.tombstone("bind-1", expected_revision=0)
    with pytest.raises(BindingLifecycleError):
        binding_service.enable("bind-1", expected_revision=1)


def test_binding_config_survives_installation_upgrade() -> None:
    installation_service, binding_service, _ = _services()
    _verified_installation(installation_service, installation_id="inst-v1")
    installation_service.promote_verified_to_active("inst-v1")
    binding_service.create_binding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        config={"mode": "fast", "top_k": 5},
    )
    _verified_installation(installation_service, installation_id="inst-v2", package_identity=_PACKAGE_B)
    installation_service.promote_verified_to_active("inst-v2", expected_active_installation_id="inst-v1")
    refreshed = binding_service.refresh_active_installation_for_slot(
        "slot-search-prod",
        prior_active_installation_id="inst-v1",
        next_active_installation_id="inst-v2",
    )
    assert len(refreshed) == 1
    binding = refreshed[0].value
    assert binding.active_installation_id == "inst-v2"
    assert binding.installation_slot_id == "slot-search-prod"
    assert dict(binding.config) == {"mode": "fast", "top_k": 5}
    resolved = binding_service.list_bindings_for_environment("env-prod")[0]
    assert resolved.active_installation_id == "inst-v2"
