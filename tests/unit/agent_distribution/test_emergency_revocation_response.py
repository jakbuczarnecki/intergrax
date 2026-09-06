# © Artur Czarnecki. All rights reserved.

"""AC-6 Phase 4 — active runtime emergency revocation response tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceState,
    FakeInMemoryRuntimeDeploymentAdapter,
)
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.emergency_revocation_response import (
    AgentEmergencyRevocationRequest,
    AgentEmergencyRevocationService,
    EmergencyTrustResponseAction,
    EmergencyTrustResponseReasonCode,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentInstallationStore,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustRevocationState,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from intergrax.core.qualification import QualificationStatus

_APP = "app-a"
_ENV = "env-prod"
_ARTIFACT_N = "sha256:" + ("a" * 64)
_ARTIFACT_N1 = "sha256:" + ("b" * 64)
_DIGEST_N = "sha256:" + ("1" * 64)
_DIGEST_N1 = "sha256:" + ("2" * 64)
_PACKAGE_ID = "intergrax-test-agent"
_FIXED_AT = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
_QUALIFIED_AT = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)
_PUBLISHER = "publisher:acme"
_EVIDENCE_N = "evidence:n"
_EVIDENCE_N1 = "evidence:n1"


def _package(digest: str) -> AgentPackageIdentity:
    return AgentPackageIdentity(
        distribution_package_id=_PACKAGE_ID,
        package_version="1.0.0",
        package_digest=digest,
    )


def _trust_record(
    digest: str,
    *,
    publisher: str = _PUBLISHER,
    evidence_id: str = _EVIDENCE_N,
    qualified_at: datetime = _QUALIFIED_AT,
) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id=evidence_id,
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
            ),
        ),
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref=publisher,
        source_provider_id="builtin-1",
        qualification_qualified_at=qualified_at,
    )


def _installation(
    installation_id: str,
    digest: str,
    *,
    slot_id: str = "slot-1",
    trust_record: AgentInstallationTrustRecord | None = None,
) -> AgentInstallationRecord:
    return AgentInstallationRecord(
        installation_id=installation_id,
        installation_slot_id=slot_id,
        environment_id=_ENV,
        package_identity=_package(digest),
        installation_state=InstallationState.INSTALLED_ACTIVE,
        active_for_slot=True,
        artifact_store_ref=f"store://artifacts/{installation_id}",
        trust_record=trust_record or _trust_record(digest),
    )


def _roster_entry(
    installation_id: str,
    digest: str,
    *,
    slot_id: str = "slot-1",
) -> EffectiveRosterEntry:
    return EffectiveRosterEntry(
        logical_agent_id="agent-1",
        installation_slot_id=slot_id,
        active_installation_id=installation_id,
        package_digest=digest,
        distribution_package_id=_PACKAGE_ID,
        effective_enablement=True,
    )


def _roster(installation_id: str, digest: str) -> EffectiveRoster:
    return EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id="rel-1",
        entries=(_roster_entry(installation_id, digest),),
    )


def _persist_roster(
    store: InMemoryEffectiveRosterSnapshotStore,
    roster: EffectiveRoster,
) -> str:
    revision_id = roster.compute_revision_id()
    persisted = roster.model_copy(update={"effective_roster_revision_id": revision_id})
    store.persist(persisted)
    return revision_id


def _revision(
    revision_id: str,
    *,
    state: RuntimeRevisionState,
    artifact: str,
    roster_revision_id: str,
    digest: str,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id=roster_revision_id,
        installed_agent_package_digests=(digest,),
        materialized_runtime_lock_id=f"lock-{revision_id}",
        materialized_runtime_lock_digest=f"lock-digest-{revision_id}",
        runtime_graph_digest=f"graph-digest-{revision_id}",
        materialization_artifact_digest=artifact,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=datetime.now(UTC) if state is RuntimeRevisionState.ACTIVE else None,
    )


@dataclass
class EmergencyHarness:
    state: AgentDistributionStoreState
    activation: ActivationService
    emergency: AgentEmergencyRevocationService
    revision_service: RuntimeRevisionService
    deployment_adapter: FakeInMemoryRuntimeDeploymentAdapter
    serving_store: InMemoryApplicationEnvironmentServingStore


def build_emergency_harness(
    *,
    n_digest: str = _DIGEST_N,
    n1_digest: str = _DIGEST_N1,
    n_trust: AgentInstallationTrustRecord | None = None,
    n1_trust: AgentInstallationTrustRecord | None = None,
) -> EmergencyHarness:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    activation_store = InMemoryApplicationEnvironmentActivationStore(state)
    roster_store = InMemoryEffectiveRosterSnapshotStore(state)
    roster_authority = EffectiveRosterAuthorityService(snapshot_store=roster_store)
    deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    projection = FakeRuntimeServingProjectionCoordinator()
    revision_service = RuntimeRevisionService(revision_store)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=activation_store,
        deployment_adapter=deployment_adapter,
        projection_coordinator=projection,
    )
    emergency = AgentEmergencyRevocationService(
        serving_store=serving_store,
        revision_store=revision_store,
        effective_roster_authority=roster_authority,
        installation_store=installation_store,
        package_trust_coordinator=AgentPackageTrustCoordinator(),
        activation_service=activation,
    )

    installation_store.persist_installation(
        _installation("inst-n", n_digest, trust_record=n_trust).model_copy(
            update={
                "active_for_slot": False,
                "installation_state": InstallationState.INSTALLED_PREVIOUS,
            }
        )
    )
    installation_store.persist_installation(
        _installation("inst-n1", n1_digest, trust_record=n1_trust)
    )
    roster_n_id = _persist_roster(roster_store, _roster("inst-n", n_digest))
    roster_n1_id = _persist_roster(roster_store, _roster("inst-n1", n1_digest))

    for revision_id, artifact, roster_id, digest in (
        ("rev-n", _ARTIFACT_N, roster_n_id, n_digest),
        ("rev-n1", _ARTIFACT_N1, roster_n1_id, n1_digest),
    ):
        candidate = _revision(
            revision_id,
            state=RuntimeRevisionState.CANDIDATE,
            artifact=artifact,
            roster_revision_id=roster_id,
            digest=digest,
        )
        revision_service.persist_candidate_revision(candidate)
        validated = _revision(
            revision_id,
            state=RuntimeRevisionState.VALIDATED,
            artifact=artifact,
            roster_revision_id=roster_id,
            digest=digest,
        )
        revision_service.mark_validated(revision_id, validated_revision=validated)

    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n",
        artifact_locator=f"artifact://{_ARTIFACT_N}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT_N,
    )
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        artifact_locator=f"artifact://{_ARTIFACT_N1}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        expected_prior_traffic_revision_id="rev-n",
        expected_serving_pointer_revision=1,
        expected_artifact_digest=_ARTIFACT_N1,
    )
    return EmergencyHarness(
        state=state,
        activation=activation,
        emergency=emergency,
        revision_service=revision_service,
        deployment_adapter=deployment_adapter,
        serving_store=serving_store,
    )


def _request(
    harness: EmergencyHarness,
    revocation: AgentPackageTrustRevocationState,
    *,
    policy: AgentPackageTrustPolicy | None = None,
    evaluated_at: datetime = _FIXED_AT,
) -> AgentEmergencyRevocationRequest:
    serving = harness.serving_store.get_serving_record(_APP, _ENV)
    assert serving is not None
    return AgentEmergencyRevocationRequest(
        application_id=_APP,
        application_environment_id=_ENV,
        expected_current_traffic_revision_id=serving.traffic_serving_revision_id,
        expected_serving_pointer_revision=serving.serving_pointer_revision,
        evaluated_at=evaluated_at,
        revocation_state=revocation,
        trust_policy=policy,
    )


def test_no_revocation_returns_no_action_without_mutations() -> None:
    harness = build_emergency_harness()
    serving_before = harness.serving_store.get_serving_record(_APP, _ENV)
    response = harness.emergency.respond_to_current_revocation(
        _request(harness, AgentPackageTrustRevocationState())
    )
    serving_after = harness.serving_store.get_serving_record(_APP, _ENV)
    assert response.action is EmergencyTrustResponseAction.NO_ACTION
    assert response.response_reason_code is EmergencyTrustResponseReasonCode.NO_ACTIVE_REVOCATION
    assert serving_before == serving_after


def test_active_digest_revoked_triggers_safe_rollback() -> None:
    harness = build_emergency_harness()
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N1}),
            ),
        )
    )
    serving = harness.serving_store.get_serving_record(_APP, _ENV)
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    )
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-n"
    assert harness.revision_service.get_active_revision(_APP, _ENV).runtime_revision_id == "rev-n"
    n1_instance = harness.activation._deployment_instance_store.get_instance(
        _APP, _ENV, "rev-n1"
    )
    assert n1_instance is not None
    assert n1_instance.instance_state is DeploymentInstanceState.DRAINING


def test_active_publisher_revoked_triggers_safe_rollback() -> None:
    harness = build_emergency_harness(
        n_trust=_trust_record(_DIGEST_N, publisher="publisher:safe"),
        n1_trust=_trust_record(_DIGEST_N1, publisher=_PUBLISHER),
    )
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(revoked_publisher_ids=frozenset({_PUBLISHER})),
        )
    )
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert response.response_reason_code is (
        EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    )


def test_active_evidence_revoked_triggers_safe_rollback() -> None:
    harness = build_emergency_harness(
        n1_trust=_trust_record(_DIGEST_N1, evidence_id=_EVIDENCE_N1),
    )
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_evidence_ids=frozenset({_EVIDENCE_N1}),
            ),
        )
    )
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert (
        response.response_reason_code
        is EmergencyTrustResponseReasonCode.SAFE_ROLLBACK_COMPLETED
    )


def test_revoked_source_does_not_trigger_emergency_response() -> None:
    harness = build_emergency_harness()
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_catalog_source_ids=frozenset({"builtin-1"}),
            ),
        )
    )
    assert response.action is EmergencyTrustResponseAction.NO_ACTION


def test_disabled_source_does_not_trigger_emergency_response() -> None:
    harness = build_emergency_harness()
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                disabled_catalog_source_ids=frozenset({"builtin-1"}),
            ),
        )
    )
    assert response.action is EmergencyTrustResponseAction.NO_ACTION


def test_prior_revision_also_revoked_blocks_rollback() -> None:
    harness = build_emergency_harness()
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N, _DIGEST_N1}),
            ),
        )
    )
    serving = harness.serving_store.get_serving_record(_APP, _ENV)
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert (
        response.response_reason_code
        is EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED
    )
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-n1"


def test_no_prior_revision_blocks_emergency_response() -> None:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    activation_store = InMemoryApplicationEnvironmentActivationStore(state)
    roster_store = InMemoryEffectiveRosterSnapshotStore(state)
    roster_authority = EffectiveRosterAuthorityService(snapshot_store=roster_store)
    deployment_adapter = FakeInMemoryRuntimeDeploymentAdapter()
    projection = FakeRuntimeServingProjectionCoordinator()
    revision_service = RuntimeRevisionService(revision_store)
    activation = ActivationService(
        revision_store=revision_store,
        deployment_instance_store=deployment_store,
        serving_store=serving_store,
        activation_store=activation_store,
        deployment_adapter=deployment_adapter,
        projection_coordinator=projection,
    )
    emergency = AgentEmergencyRevocationService(
        serving_store=serving_store,
        revision_store=revision_store,
        effective_roster_authority=roster_authority,
        installation_store=installation_store,
        package_trust_coordinator=AgentPackageTrustCoordinator(),
        activation_service=activation,
    )
    installation_store.persist_installation(_installation("inst-n1", _DIGEST_N1))
    roster_id = _persist_roster(roster_store, _roster("inst-n1", _DIGEST_N1))
    candidate = _revision(
        "rev-n1",
        state=RuntimeRevisionState.CANDIDATE,
        artifact=_ARTIFACT_N1,
        roster_revision_id=roster_id,
        digest=_DIGEST_N1,
    )
    revision_service.persist_candidate_revision(candidate)
    validated = _revision(
        "rev-n1",
        state=RuntimeRevisionState.VALIDATED,
        artifact=_ARTIFACT_N1,
        roster_revision_id=roster_id,
        digest=_DIGEST_N1,
    )
    revision_service.mark_validated("rev-n1", validated_revision=validated)
    activation.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        artifact_locator=f"artifact://{_ARTIFACT_N1}",
    )
    activation.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        expected_prior_traffic_revision_id=None,
        expected_serving_pointer_revision=0,
        expected_artifact_digest=_ARTIFACT_N1,
    )
    response = emergency.respond_to_current_revocation(
        AgentEmergencyRevocationRequest(
            application_id=_APP,
            application_environment_id=_ENV,
            evaluated_at=_FIXED_AT,
            revocation_state=AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N1}),
            ),
        )
    )
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert response.response_reason_code is EmergencyTrustResponseReasonCode.NO_PRIOR_REVISION


def test_stale_prior_qualification_blocks_rollback_target() -> None:
    harness = build_emergency_harness(
        n_trust=_trust_record(
            _DIGEST_N,
            qualified_at=_QUALIFIED_AT - timedelta(days=30),
        ),
    )
    policy = AgentPackageTrustPolicy(
        posture=AgentPackageTrustPosture.PRODUCTION,
        max_qualification_age=timedelta(days=7),
    )
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N1}),
            ),
            policy=policy,
        )
    )
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert (
        response.response_reason_code
        is EmergencyTrustResponseReasonCode.ROLLBACK_TARGET_UNTRUSTED
    )


def test_expired_qualification_on_active_does_not_trigger_emergency() -> None:
    harness = build_emergency_harness(
        n1_trust=_trust_record(
            _DIGEST_N1,
            qualified_at=_QUALIFIED_AT - timedelta(days=30),
        ),
    )
    policy = AgentPackageTrustPolicy(
        posture=AgentPackageTrustPosture.PRODUCTION,
        max_qualification_age=timedelta(days=7),
    )
    response = harness.emergency.respond_to_current_revocation(
        _request(harness, AgentPackageTrustRevocationState(), policy=policy)
    )
    assert response.action is EmergencyTrustResponseAction.NO_ACTION


def test_installed_inactive_revoked_package_is_no_action() -> None:
    harness = build_emergency_harness()
    inactive = _installation("inst-inactive", _DIGEST_N)
    harness.state.installations["inst-inactive"] = inactive
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N}),
            ),
        )
    )
    assert response.action is EmergencyTrustResponseAction.NO_ACTION


def test_stale_expected_serving_pointer_fails_closed() -> None:
    harness = build_emergency_harness()
    request = _request(
        harness,
        AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({_DIGEST_N1}),
        ),
    )
    stale = request.model_copy(update={"expected_current_traffic_revision_id": "rev-n"})
    response = harness.emergency.respond_to_current_revocation(stale)
    assert response.action is EmergencyTrustResponseAction.BLOCKED_NO_SAFE_TARGET
    assert (
        response.response_reason_code
        is EmergencyTrustResponseReasonCode.SERVING_POINTER_MISMATCH
    )


def test_idempotent_retry_after_successful_rollback() -> None:
    harness = build_emergency_harness()
    request = _request(
        harness,
        AgentPackageTrustRevocationState(
            revoked_package_digests=frozenset({_DIGEST_N1}),
        ),
    )
    first = harness.emergency.respond_to_current_revocation(request)
    assert first.action is EmergencyTrustResponseAction.ROLLBACK
    second = harness.emergency.respond_to_current_revocation(request)
    assert second.action is EmergencyTrustResponseAction.NO_ACTION
    assert (
        second.response_reason_code
        is EmergencyTrustResponseReasonCode.NO_ACTIVE_REVOCATION
    )


def test_drain_start_failure_reports_recovery_required() -> None:
    harness = build_emergency_harness()
    instance = harness.activation._deployment_instance_store.get_instance(
        _APP, _ENV, "rev-n1"
    )
    assert instance is not None and instance.serving_unit_ref is not None
    harness.deployment_adapter.fail_begin_drain(instance.serving_unit_ref)
    response = harness.emergency.respond_to_current_revocation(
        _request(
            harness,
            AgentPackageTrustRevocationState(
                revoked_package_digests=frozenset({_DIGEST_N1}),
            ),
        )
    )
    serving = harness.serving_store.get_serving_record(_APP, _ENV)
    assert response.action is EmergencyTrustResponseAction.ROLLBACK
    assert (
        response.response_reason_code
        is EmergencyTrustResponseReasonCode.DRAIN_RECOVERY_REQUIRED
    )
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-n"
