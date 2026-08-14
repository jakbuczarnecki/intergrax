# © Artur Czarnecki. All rights reserved.

"""AP-6 installed-agent requirement set and candidate dependency specification tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    InstalledAgentPackageRequirement,
    InstalledAgentRequirementSet,
    PolicyDependencyConstraint,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.dependency_specification import (
    build_candidate_dependency_specification,
)
from intergrax.agent_distribution.effective_roster import (
    EffectiveRosterBuilder,
    InstalledAgentRequirementSetBuilder,
)
from intergrax.agent_distribution.errors import DependencySpecificationError
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
)
from intergrax.agent_distribution.installation import (
    AgentInstallationRecord,
    InstallationState,
)
from intergrax.agent_distribution.installation_slot_scope import InstallationSlotScope
from intergrax.agent_distribution.roster import (
    EffectiveRosterEntry,
    ManifestDefaultAgentDeclaration,
)
from intergrax.agent_distribution.stores import AgentArtifactMetadata
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentQualificationStatus,
    AgentTrustEvidenceRef,
)

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_APP_ID = "app-local-workspace"
_ENV_ID = "env-prod"
_RELEASE_ID = "rel-2026-08-13"
_PACKAGE_A = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST_A,
)


def _trust_record(digest: str = _DIGEST_A) -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=digest,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _metadata(
    *,
    digest: str = _DIGEST_A,
    distribution_package_id: str = "intergrax-local-search-agent",
    metadata_ref: str = "meta://search/pyproject.toml",
) -> AgentArtifactMetadata:
    return AgentArtifactMetadata(
        package_digest=digest,
        artifact_store_ref=f"store://artifacts/{digest}",
        distribution_package_id=distribution_package_id,
        agent_project_metadata_ref=metadata_ref,
    )


def test_effective_roster_to_digest_pinned_requirement_set() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata())
    state.installations["inst-v1"] = AgentInstallationRecord(
        installation_id="inst-v1",
        installation_slot_id="slot-search-prod",
        environment_id=_ENV_ID,
        package_identity=_PACKAGE_A,
        installation_state=InstallationState.INSTALLED_ACTIVE,
        active_for_slot=True,
        artifact_store_ref="store://artifacts/inst-v1",
        trust_record=_trust_record(),
    )
    state.active_installation_by_scope[
        InstallationSlotScope(environment_id=_ENV_ID, installation_slot_id="slot-search-prod")
    ] = "inst-v1"
    roster = EffectiveRosterBuilder(InMemoryAgentInstallationStore(state)).build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(
            ManifestDefaultAgentDeclaration(
                logical_agent_id="search",
                installation_slot_id="slot-search-prod",
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST_A,
                builtin_package_ref="builtin:intergrax-local-search-agent",
            ),
        ),
        durable_bindings=(),
    )
    requirement_set = InstalledAgentRequirementSetBuilder(metadata_store).build(roster)
    assert len(requirement_set.agent_packages) == 1
    requirement = requirement_set.agent_packages[0]
    assert requirement.package_digest == _DIGEST_A
    assert requirement.agent_project_metadata_ref == "meta://search/pyproject.toml"


def test_duplicate_package_requirements_deduplicate_deterministically() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata())
    roster = (
        EffectiveRosterBuilder(InMemoryAgentInstallationStore(state))
        .build(
            application_id=_APP_ID,
            application_environment_id=_ENV_ID,
            manifest_release_id=_RELEASE_ID,
            manifest_defaults=(),
            durable_bindings=(),
        )
        .model_copy(
            update={
                "entries": (
                    EffectiveRosterEntry(
                        logical_agent_id="search-a",
                        installation_slot_id="slot-a",
                        package_digest=_DIGEST_A,
                        distribution_package_id="intergrax-local-search-agent",
                        effective_enablement=True,
                    ),
                    EffectiveRosterEntry(
                        logical_agent_id="search-b",
                        installation_slot_id="slot-b",
                        package_digest=_DIGEST_A,
                        distribution_package_id="intergrax-local-search-agent",
                        effective_enablement=True,
                    ),
                )
            }
        )
        .with_revision_id()
    )
    requirement_set = InstalledAgentRequirementSetBuilder(metadata_store).build(roster)
    assert len(requirement_set.agent_packages) == 1


def test_conflicting_digest_for_same_distribution_package_fails_closed() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata(digest=_DIGEST_A))
    metadata_store.persist_metadata(
        _metadata(digest=_DIGEST_B, metadata_ref="meta://search/v2/pyproject.toml")
    )
    roster = (
        EffectiveRosterBuilder(InMemoryAgentInstallationStore(state))
        .build(
            application_id=_APP_ID,
            application_environment_id=_ENV_ID,
            manifest_release_id=_RELEASE_ID,
            manifest_defaults=(),
            durable_bindings=(),
        )
        .model_copy(
            update={
                "entries": (
                    EffectiveRosterEntry(
                        logical_agent_id="search-a",
                        installation_slot_id="slot-a",
                        package_digest=_DIGEST_A,
                        distribution_package_id="intergrax-local-search-agent",
                        effective_enablement=True,
                    ),
                    EffectiveRosterEntry(
                        logical_agent_id="search-b",
                        installation_slot_id="slot-b",
                        package_digest=_DIGEST_B,
                        distribution_package_id="intergrax-local-search-agent",
                        effective_enablement=True,
                    ),
                )
            }
        )
        .with_revision_id()
    )
    with pytest.raises(DependencySpecificationError):
        InstalledAgentRequirementSetBuilder(metadata_store).build(roster)


def test_requirement_set_uses_authoritative_metadata_ref_only() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata())
    roster = EffectiveRosterBuilder(InMemoryAgentInstallationStore(state)).build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(
            ManifestDefaultAgentDeclaration(
                logical_agent_id="search",
                installation_slot_id="slot-search-prod",
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST_A,
                builtin_package_ref="builtin:intergrax-local-search-agent",
            ),
        ),
        durable_bindings=(),
    )
    requirement = (
        InstalledAgentRequirementSetBuilder(metadata_store)
        .build(roster)
        .agent_packages[0]
    )
    assert requirement.agent_project_metadata_ref.startswith("meta://")


def test_requirement_ordering_is_deterministic() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata())
    metadata_store.persist_metadata(
        _metadata(
            digest=_DIGEST_B,
            distribution_package_id="intergrax-local-writer-agent",
            metadata_ref="meta://writer/pyproject.toml",
        )
    )
    roster = (
        EffectiveRosterBuilder(InMemoryAgentInstallationStore(state))
        .build(
            application_id=_APP_ID,
            application_environment_id=_ENV_ID,
            manifest_release_id=_RELEASE_ID,
            manifest_defaults=(),
            durable_bindings=(),
        )
        .model_copy(
            update={
                "entries": (
                    EffectiveRosterEntry(
                        logical_agent_id="writer",
                        installation_slot_id="slot-writer",
                        package_digest=_DIGEST_B,
                        distribution_package_id="intergrax-local-writer-agent",
                        effective_enablement=True,
                    ),
                    EffectiveRosterEntry(
                        logical_agent_id="search",
                        installation_slot_id="slot-search",
                        package_digest=_DIGEST_A,
                        distribution_package_id="intergrax-local-search-agent",
                        effective_enablement=True,
                    ),
                )
            }
        )
        .with_revision_id()
    )
    packages = (
        InstalledAgentRequirementSetBuilder(metadata_store).build(roster).agent_packages
    )
    assert [item.distribution_package_id for item in packages] == [
        "intergrax-local-search-agent",
        "intergrax-local-writer-agent",
    ]


def test_build_candidate_dependency_specification_from_l1_and_l2() -> None:
    state = AgentDistributionStoreState()
    metadata_store = InMemoryAgentArtifactMetadataStore(state)
    metadata_store.persist_metadata(_metadata())
    roster = EffectiveRosterBuilder(InMemoryAgentInstallationStore(state)).build(
        application_id=_APP_ID,
        application_environment_id=_ENV_ID,
        manifest_release_id=_RELEASE_ID,
        manifest_defaults=(
            ManifestDefaultAgentDeclaration(
                logical_agent_id="search",
                installation_slot_id="slot-search-prod",
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST_A,
                builtin_package_ref="builtin:intergrax-local-search-agent",
            ),
        ),
        durable_bindings=(),
    )
    installed = InstalledAgentRequirementSetBuilder(metadata_store).build(roster)
    repository = RepositoryDependencyDeclaration(
        application_release_id="rel-2026-08-13",
        direct_dependencies=("httpx", "pydantic"),
    )
    spec = build_candidate_dependency_specification(
        repository_declaration=repository,
        installed_agent_requirement_set=installed,
        platform_version="0.1.0",
        platform_extras=("dev", "otel"),
        policy_constraints=(
            PolicyDependencyConstraint(
                constraint_kind="python", constraint_value=">=3.12"
            ),
        ),
        repository_lock_hint_ref="lock://uv/workspace",
    )
    assert spec.application_release_id == repository.application_release_id
    assert spec.agent_packages == installed.agent_packages
    assert spec.platform_extras == ("dev", "otel")


def test_repository_application_release_mismatch_fails_closed() -> None:
    with pytest.raises(ValidationError):
        CandidateDependencySpecification(
            application_release_id="rel-a",
            platform_version="0.1.0",
            repository_declaration=RepositoryDependencyDeclaration(
                application_release_id="rel-b",
            ),
            agent_packages=(),
        )


def test_platform_extras_ordering_is_deterministic() -> None:
    repository = RepositoryDependencyDeclaration(application_release_id="rel-1")
    installed = InstalledAgentRequirementSet(
        effective_roster_revision_id="sha256:" + ("c" * 64),
        agent_packages=(),
    )
    spec = build_candidate_dependency_specification(
        repository_declaration=repository,
        installed_agent_requirement_set=installed,
        platform_version="0.1.0",
        platform_extras=("z-extra", "a-extra", "a-extra"),
    )
    assert spec.platform_extras == ("a-extra", "z-extra")


def test_policy_constraint_ordering_is_deterministic() -> None:
    repository = RepositoryDependencyDeclaration(application_release_id="rel-1")
    installed = InstalledAgentRequirementSet(
        effective_roster_revision_id="sha256:" + ("c" * 64),
        agent_packages=(),
    )
    spec = build_candidate_dependency_specification(
        repository_declaration=repository,
        installed_agent_requirement_set=installed,
        platform_version="0.1.0",
        policy_constraints=(
            PolicyDependencyConstraint(
                constraint_kind="deny", constraint_value="pkg-z"
            ),
            PolicyDependencyConstraint(
                constraint_kind="deny", constraint_value="pkg-a"
            ),
            PolicyDependencyConstraint(
                constraint_kind="allow", constraint_value="pkg-a"
            ),
        ),
    )
    assert [item.constraint_value for item in spec.policy_constraints] == [
        "pkg-a",
        "pkg-a",
        "pkg-z",
    ]


def test_repeated_equivalent_input_produces_identical_specification() -> None:
    repository = RepositoryDependencyDeclaration(application_release_id="rel-1")
    installed = InstalledAgentRequirementSet(
        effective_roster_revision_id="sha256:" + ("c" * 64),
        agent_packages=(
            InstalledAgentPackageRequirement(
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST_A,
                agent_project_metadata_ref="meta://search/pyproject.toml",
            ),
        ),
    )
    kwargs = {
        "repository_declaration": repository,
        "installed_agent_requirement_set": installed,
        "platform_version": "0.1.0",
        "platform_extras": ("dev",),
        "policy_constraints": (
            PolicyDependencyConstraint(
                constraint_kind="python", constraint_value=">=3.12"
            ),
        ),
    }
    first = build_candidate_dependency_specification(**kwargs)
    second = build_candidate_dependency_specification(**kwargs)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
