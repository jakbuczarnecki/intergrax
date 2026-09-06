# © Artur Czarnecki. All rights reserved.

"""Reusable AC-6 Phase 5 trust + lifecycle E2E composition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.agent_distribution.activation import (
    ActivationService,
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_service import AgentPlatformAdminService
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.agent_distribution.deployment import FakeInMemoryRuntimeDeploymentAdapter
from intergrax.agent_distribution.effective_roster_authority import (
    EffectiveRosterAuthorityService,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.materialization_service import (
    RuntimeMaterializationService,
)
from intergrax.agent_distribution.package_trust import AgentPackageTrustCoordinator
from intergrax.agent_distribution.runtime_graph_service import (
    CandidateRuntimeGraphBuilder,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.runtime_revision_service import RuntimeRevisionService
from intergrax.agent_distribution.trust import (
    AgentDeliverySource,
    AgentInstallationTrustRecord,
    AgentPackageQualificationResult,
    AgentPackageTrustDecision,
    AgentPackageTrustPolicy,
    AgentPackageTrustPosture,
    AgentPackageTrustRevocationState,
    AgentPublisherIdentity,
    AgentQualificationEvidenceKind,
)
from intergrax.agent_distribution.binding_service import BindingService
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.installation_service import InstallationService
from intergrax.agent_distribution.agent_project_metadata import AgentProjectMetadata
from intergrax.core.qualification import QualificationEvidence, QualificationStatus
from testing_support.agent_package_attestation import (
    build_test_attestation_trust_coordinator,
    verified_signature_qualification_evidence,
)
from testing_support.agent_platform_dependency_resolver import (
    make_identity_dependency_resolver,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    _DeterministicAdapter,
    _FakeCatalog,
    _MetadataProvider,
    admin_test_principal,
    allow_mutation_boundary,
)

AC6_APP = "app-ac6"
AC6_ENV = "env-prod"
AC6_PACKAGE_ID = "intergrax-local-search-agent"
AC6_META_REF = "meta://ac6-search"
AC6_SLOT = "slot-search"
AC6_BINDING = "bind-search"
AC6_PUBLISHER_ID = "publisher:acme"
AC6_KEY_ID = "test-publisher-key-1"
AC6_SOURCE_ID = "builtin-1"
AC6_ARTIFACT = "sha256:" + ("d" * 64)
AC6_DIGEST_D1 = "sha256:" + ("1" * 64)
AC6_DIGEST_D2 = "sha256:" + ("2" * 64)
AC6_FIXED_AT = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
AC6_QUALIFIED_AT = datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)
AC6_EVAL_FRESH = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)
AC6_EVAL_STALE = datetime(2026, 8, 20, 12, 0, 0, tzinfo=UTC)

AC6_PUBLISHER = AgentPublisherIdentity(
    publisher_id=AC6_PUBLISHER_ID,
    display_name="ACME",
)
AC6_CATALOG_SOURCE = CatalogSourceIdentity(
    catalog_source_id=AC6_SOURCE_ID,
    provider_kind=CatalogProviderKind.BUILTIN,
)


def ac6_package_identity(
    digest: str, *, version: str = "1.0.0"
) -> AgentPackageIdentity:
    return AgentPackageIdentity(
        distribution_package_id=AC6_PACKAGE_ID,
        package_version=version,
        package_digest=digest,
    )


def ac6_production_policy(**overrides: object) -> AgentPackageTrustPolicy:
    base = {
        "posture": AgentPackageTrustPosture.PRODUCTION,
        "trust_profile_ref": "profile:production",
        "permitted_provider_kinds": frozenset(
            {CatalogProviderKind.BUILTIN, CatalogProviderKind.OFFICIAL_CATALOG}
        ),
        "permitted_delivery_sources": frozenset(
            {AgentDeliverySource.BUILTIN, AgentDeliverySource.MARKETPLACE}
        ),
        "required_evidence_kinds": frozenset(
            {
                AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
                AgentQualificationEvidenceKind.REVOCATION_CHECK,
            }
        ),
    }
    base.update(overrides)
    return AgentPackageTrustPolicy(**base)


def ac6_qualification(
    package_identity: AgentPackageIdentity,
    *,
    qualified_at: datetime = AC6_QUALIFIED_AT,
    signature_b64: str | None = None,
    public_key_bytes: bytes | None = None,
) -> AgentPackageQualificationResult:
    if signature_b64 is None or public_key_bytes is None:
        signature_evidence = verified_signature_qualification_evidence(
            package_identity=package_identity,
            publisher_id=AC6_PUBLISHER_ID,
            key_id=AC6_KEY_ID,
        )
    else:
        from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
            Ed25519PackageAttestationVerifier,
        )
        from intergrax.agent_distribution.package_attestation import (
            AgentPackageAttestationAlgorithm,
            AgentPackageAttestationVerificationRequest,
            StaticPublisherVerificationKeyProvider,
        )

        verifier = Ed25519PackageAttestationVerifier(
            key_provider=StaticPublisherVerificationKeyProvider(
                {(AC6_PUBLISHER_ID, AC6_KEY_ID): public_key_bytes}
            )
        )
        signature_evidence = verifier.verify_qualification_evidence(
            AgentPackageAttestationVerificationRequest(
                package_identity=package_identity,
                publisher_id=AC6_PUBLISHER_ID,
                attestation_id="attest-ac6",
                key_id=AC6_KEY_ID,
                algorithm=AgentPackageAttestationAlgorithm.ED25519,
                signature_b64=signature_b64,
            )
        )
    return AgentPackageQualificationResult(
        publisher=AC6_PUBLISHER,
        status=QualificationStatus.PRODUCTION_QUALIFIED,
        evidence=(
            signature_evidence,
            QualificationEvidence(
                kind=AgentQualificationEvidenceKind.REVOCATION_CHECK,
                code="revocation_ok",
                ref="rev-ref",
            ),
        ),
        reason="qualified by AC-6 E2E evidence",
        delivery_source=AgentDeliverySource.BUILTIN,
        qualified_at=qualified_at,
    )


def ac6_evaluate_trust(
    coordinator: AgentPackageTrustCoordinator,
    package_identity: AgentPackageIdentity,
    *,
    qualification: AgentPackageQualificationResult,
    revocation_state: AgentPackageTrustRevocationState | None = None,
    policy: AgentPackageTrustPolicy | None = None,
    evaluated_at: datetime = AC6_FIXED_AT,
) -> AgentPackageTrustDecision:
    return coordinator.evaluate(
        package_identity=package_identity,
        catalog_source=AC6_CATALOG_SOURCE,
        delivery_source=AgentDeliverySource.BUILTIN,
        publisher=AC6_PUBLISHER,
        policy=policy or ac6_production_policy(),
        qualification=qualification,
        evidence_package_digest=package_identity.package_digest,
        evidence_id="evidence:ac6",
        evaluated_at=evaluated_at,
        revocation_state=revocation_state or AgentPackageTrustRevocationState(),
    )


@dataclass
class Ac6AdminStack:
    service: AgentPlatformAdminService
    state: AgentDistributionStoreState
    coordinator: AgentPackageTrustCoordinator
    evaluation_times: dict[str, datetime]
    revocation_state: dict[str, AgentPackageTrustRevocationState]


def build_ac6_admin_stack(
    *,
    evaluation_at: datetime = AC6_EVAL_FRESH,
    revocation_state: AgentPackageTrustRevocationState | None = None,
    policy: AgentPackageTrustPolicy | None = None,
    coordinator: AgentPackageTrustCoordinator | None = None,
) -> Ac6AdminStack:
    state = AgentDistributionStoreState()
    installation_store = InMemoryAgentInstallationStore(state)
    binding_store = InMemoryApplicationAgentBindingStore(state)
    revision_store = InMemoryRuntimeRevisionStore(state)
    serving_store = InMemoryApplicationEnvironmentServingStore(state)
    deployment_store = InMemoryDeploymentInstanceStore(state)
    lock_store = InMemoryMaterializedRuntimeLockStore(state)
    materialization_store = InMemoryRuntimeMaterializationStore(state)
    effective_roster_snapshot_store = InMemoryEffectiveRosterSnapshotStore(state)
    effective_roster_authority = EffectiveRosterAuthorityService(
        snapshot_store=effective_roster_snapshot_store,
    )
    artifact_store = InMemoryAgentArtifactMetadataStore(state)
    installation_service = InstallationService(installation_store)
    binding_service = BindingService(binding_store, installation_service)
    revision_service = RuntimeRevisionService(revision_store)
    metadata_provider = _MetadataProvider(
        {
            AC6_META_REF: AgentProjectMetadata(
                distribution_package_id=AC6_PACKAGE_ID,
                dependencies=(),
            )
        }
    )
    catalog = _FakeCatalog(
        [
            AgentCatalogEntry(
                catalog_entry_id="cat-researcher",
                catalog_source=AC6_CATALOG_SOURCE,
                display_name="Researcher",
                package_id_line=AC6_PACKAGE_ID,
            )
        ]
    )
    trust_coordinator = coordinator or build_test_attestation_trust_coordinator()
    evaluation_times: dict[str, datetime] = {"at": evaluation_at}
    revocation_holder: dict[str, AgentPackageTrustRevocationState] = {
        "state": revocation_state or AgentPackageTrustRevocationState(),
    }
    policy_holder: dict[str, AgentPackageTrustPolicy] = {
        "policy": policy or ac6_production_policy(),
    }
    service = AgentPlatformAdminService(
        installation_store=installation_store,
        binding_store=binding_store,
        revision_store=revision_store,
        serving_store=serving_store,
        deployment_instance_store=deployment_store,
        lock_store=lock_store,
        materialization_store=materialization_store,
        effective_roster_snapshot_store=effective_roster_snapshot_store,
        effective_roster_authority=effective_roster_authority,
        artifact_metadata_store=artifact_store,
        installation_service=installation_service,
        binding_service=binding_service,
        revision_service=revision_service,
        roster_builder=EffectiveRosterBuilder(installation_store),
        requirement_set_builder=InstalledAgentRequirementSetBuilder(artifact_store),
        activation_service=ActivationService(
            revision_store=revision_store,
            deployment_instance_store=deployment_store,
            serving_store=serving_store,
            activation_store=InMemoryApplicationEnvironmentActivationStore(state),
            deployment_adapter=FakeInMemoryRuntimeDeploymentAdapter(),
            projection_coordinator=FakeRuntimeServingProjectionCoordinator(),
        ),
        graph_builder=CandidateRuntimeGraphBuilder(metadata_provider),
        materialization_service=RuntimeMaterializationService(
            {MaterializationTopology.OCI_IMAGE: _DeterministicAdapter()}
        ),
        metadata_provider=metadata_provider,
        catalog_provider=catalog,
        dependency_resolver=make_identity_dependency_resolver(),
        mutation_authorization_boundary=allow_mutation_boundary(),
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            "tenant-test"
        ),
        package_trust_coordinator=trust_coordinator,
        package_trust_revocation_state_source=lambda: revocation_holder["state"],
        package_trust_policy_source=lambda: policy_holder["policy"],
        package_trust_evaluation_time_source=lambda: evaluation_times["at"],
    )
    return Ac6AdminStack(
        service=service,
        state=state,
        coordinator=trust_coordinator,
        evaluation_times=evaluation_times,
        revocation_state=revocation_holder,
    )


def ac6_require_trust_record(
    decision: AgentPackageTrustDecision,
) -> AgentInstallationTrustRecord:
    assert decision.trust_record is not None
    return decision.trust_record


def ac6_admin_principal():
    return admin_test_principal()
