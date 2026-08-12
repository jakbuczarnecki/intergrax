# © Artur Czarnecki. All rights reserved.

"""AP-4 concurrency and atomicity tests."""

from __future__ import annotations

import threading

import pytest

from intergrax.agent_distribution.errors import InstallationSlotConflict
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.runtime_revision import MaterializationTopology, RuntimeRevision, RuntimeRevisionState
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationStatus,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)


def _trust_record() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
    )


def _prepare_verified(
    installation_service: InstallationService,
    installation_id: str,
    digest: str,
) -> None:
    installation_service.create_candidate_installation(
        installation_id=installation_id,
        installation_slot_id="slot-search-prod",
        environment_id="env-prod",
        package_identity=AgentPackageIdentity(
            distribution_package_id="intergrax-local-search-agent",
            package_version=installation_id,
            package_digest=digest,
        ),
    )
    installation_service.mark_verified(
        installation_id,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=_trust_record(),
    )


def test_concurrent_install_on_same_slot_one_succeeds_one_conflicts() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryAgentInstallationStore(state)
    service = InstallationService(store)
    _prepare_verified(service, "inst-v0", _DIGEST_A)
    _prepare_verified(service, "inst-v1", _DIGEST_B)
    _prepare_verified(service, "inst-v2", "sha256:" + ("c" * 64))
    service.promote_verified_to_active("inst-v0")
    results: list[str] = []
    errors: list[Exception] = []

    def promote(installation_id: str) -> None:
        try:
            service.promote_verified_to_active(
                installation_id,
                expected_active_installation_id="inst-v0",
            )
            results.append(installation_id)
        except InstallationSlotConflict as exc:
            errors.append(exc)

    first = threading.Thread(target=promote, args=("inst-v1",))
    second = threading.Thread(target=promote, args=("inst-v2",))
    first.start()
    second.start()
    first.join()
    second.join()
    assert len(results) == 1
    assert len(errors) == 1
    active = service.resolve_active_for_slot("slot-search-prod")
    assert active is not None
    assert active.installation_id in {"inst-v1", "inst-v2"}


def test_partial_installation_promotion_failure_leaves_no_double_active() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryAgentInstallationStore(state)
    store._fail_after_prior_demotion = True
    service = InstallationService(store)
    _prepare_verified(service, "inst-a", _DIGEST_A)
    _prepare_verified(service, "inst-b", _DIGEST_B)
    service.promote_verified_to_active("inst-a")
    with pytest.raises(InstallationSlotConflict):
        service.promote_verified_to_active("inst-b", expected_active_installation_id="inst-a")
    active = service.resolve_active_for_slot("slot-search-prod")
    assert active is not None
    assert active.installation_id == "inst-a"
    assert state.installations["inst-a"].installation_state.value == "installed_active"
    assert state.installations["inst-b"].installation_state.value == "verified"


def test_partial_runtime_revision_activation_failure_leaves_prior_active() -> None:
    state = AgentDistributionStoreState()
    store = InMemoryRuntimeRevisionStore(state)
    store._fail_after_prior_supersede = True
    service = RuntimeRevisionService(store)

    def candidate(revision_id: str) -> RuntimeRevision:
        return RuntimeRevision(
            runtime_revision_id=revision_id,
            application_environment_id="env-prod",
            application_release_id="rel-1",
            platform_version="0.1.0",
            effective_roster_revision_id="roster-hash",
            materialized_runtime_lock_id="lock-1",
            materialized_runtime_lock_digest="lock-digest",
            runtime_graph_digest="graph-digest",
            materialization_artifact_digest="artifact-digest",
            materialization_topology=MaterializationTopology.VENV_BUNDLE,
            revision_state=RuntimeRevisionState.CANDIDATE,
        )

    service.persist_candidate_revision(candidate("rev-1"))
    service.mark_validated("rev-1", validated_revision=candidate("rev-1").model_copy(
        update={"revision_state": RuntimeRevisionState.VALIDATED}
    ))
    service.activate_revision("rev-1")
    service.persist_candidate_revision(candidate("rev-2"))
    service.mark_validated("rev-2", validated_revision=candidate("rev-2").model_copy(
        update={"revision_state": RuntimeRevisionState.VALIDATED}
    ))
    from intergrax.agent_distribution.errors import RuntimeRevisionConflict

    with pytest.raises(RuntimeRevisionConflict):
        service.activate_revision("rev-2", expected_prior_active_revision_id="rev-1")
    active = service.get_active_revision("env-prod")
    assert active is not None
    assert active.runtime_revision_id == "rev-1"
    assert state.revisions["rev-1"].revision_state is RuntimeRevisionState.ACTIVE
