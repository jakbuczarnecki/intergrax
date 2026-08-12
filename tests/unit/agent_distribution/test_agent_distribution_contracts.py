# © Artur Czarnecki. All rights reserved.

"""Unit tests for Agent Distribution Tier-0 contracts (AP-3)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.agent_distribution.binding import ApplicationAgentBinding
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dependency import (
    CandidateDependencySpecification,
    DependencyResolverInput,
    InstalledAgentPackageRequirement,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
    RepositoryDependencyDeclaration,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate, AgentPackageIdentity
from intergrax.agent_distribution.installation import AgentInstallationRecord, InstallationState
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import AgentArtifactMetadata, RuntimeRevisionStore
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationStatus,
)

_DIGEST = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_PACKAGE = AgentPackageIdentity(
    distribution_package_id="intergrax-local-search-agent",
    package_version="1.0.0",
    package_digest=_DIGEST,
)


def _trust_record() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=AgentQualificationStatus.PRODUCTION_QUALIFIED,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
    )


def test_package_identity_rejects_blank_digest() -> None:
    with pytest.raises(ValidationError):
        AgentPackageIdentity(
            distribution_package_id="intergrax-local-search-agent",
            package_version="1.0.0",
            package_digest="   ",
        )


def test_stable_vs_revision_identity_separation() -> None:
    candidate = AgentPackageCandidate(
        distribution_package_id="intergrax-local-search-agent",
        package_version="1.0.0",
    )
    assert candidate.package_digest is None
    with pytest.raises(ValueError):
        candidate.to_digest_pinned()
    candidate_with_digest = candidate.model_copy(update={"package_digest": _DIGEST})
    pinned = candidate_with_digest.to_digest_pinned()
    pinned_b = _PACKAGE.model_copy(update={"package_digest": _DIGEST_B})
    assert pinned.package_digest != pinned_b.package_digest
    assert pinned.distribution_package_id == pinned_b.distribution_package_id


def test_catalog_entry_strict_extra_rejection() -> None:
    source = CatalogSourceIdentity(
        catalog_source_id="builtin",
        provider_kind=CatalogProviderKind.BUILTIN,
    )
    with pytest.raises(ValidationError):
        AgentCatalogEntry(
            catalog_entry_id="entry-1",
            catalog_source=source,
            display_name="Search",
            package_id_line="intergrax-local-search-agent",
            unexpected_field=True,
        )


def test_binding_survives_installation_upgrade_via_slot() -> None:
    binding = ApplicationAgentBinding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        active_installation_id="inst-v1",
        binding_revision=1,
    )
    assert binding.survives_installation_upgrade(
        prior_active_installation_id="inst-v1",
        next_active_installation_id="inst-v2",
    )
    upgraded = binding.model_copy(update={"active_installation_id": "inst-v2", "binding_revision": 2})
    assert upgraded.installation_slot_id == "slot-search-prod"
    assert upgraded.logical_agent_id == "search"


def test_binding_config_rejects_secret_values() -> None:
    with pytest.raises(ValidationError):
        ApplicationAgentBinding(
            application_binding_id="bind-1",
            application_id="demo_app",
            application_environment_id="env-prod",
            logical_agent_id="search",
            installation_slot_id="slot-search-prod",
            config={"api_key": "sk-live-not-a-ref"},
            binding_revision=0,
        )


def test_binding_config_rejects_nested_secret_key() -> None:
    with pytest.raises(ValidationError):
        ApplicationAgentBinding(
            application_binding_id="bind-1",
            application_id="demo_app",
            application_environment_id="env-prod",
            logical_agent_id="search",
            installation_slot_id="slot-search-prod",
            config={"provider": {"api_key": "public-value"}},
            binding_revision=0,
        )


def test_binding_config_rejects_nested_secret_value() -> None:
    with pytest.raises(ValidationError):
        ApplicationAgentBinding(
            application_binding_id="bind-1",
            application_id="demo_app",
            application_environment_id="env-prod",
            logical_agent_id="search",
            installation_slot_id="slot-search-prod",
            config={"provider": {"token_ref_name": "sk-live-not-a-ref"}},
            binding_revision=0,
        )


def test_binding_config_accepts_legitimate_nested_config() -> None:
    binding = ApplicationAgentBinding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        config={"provider": {"region": "eu-west-1", "limits": {"rpm": 120}}},
        binding_revision=0,
    )
    assert binding.config["provider"]["region"] == "eu-west-1"


def test_binding_config_is_deeply_immutable() -> None:
    binding = ApplicationAgentBinding(
        application_binding_id="bind-1",
        application_id="demo_app",
        application_environment_id="env-prod",
        logical_agent_id="search",
        installation_slot_id="slot-search-prod",
        config={"provider": {"region": "eu-west-1"}},
        binding_revision=0,
    )
    with pytest.raises(TypeError):
        binding.config["provider"]["region"] = "mutated"


def test_effective_roster_merged_config_is_deeply_immutable() -> None:
    entry = EffectiveRosterEntry(
        logical_agent_id="search",
        installation_slot_id="slot-1",
        package_digest=_DIGEST,
        distribution_package_id="intergrax-local-search-agent",
        effective_enablement=True,
        merged_config={"limits": {"rpm": 100}},
    )
    with pytest.raises(TypeError):
        entry.merged_config["limits"]["rpm"] = 999


def test_effective_roster_revision_immune_to_nested_mutation_attempt() -> None:
    entry = EffectiveRosterEntry(
        logical_agent_id="search",
        installation_slot_id="slot-1",
        package_digest=_DIGEST,
        distribution_package_id="intergrax-local-search-agent",
        effective_enablement=True,
        merged_config={"limits": {"rpm": 100}},
    )
    roster = EffectiveRoster(
        application_id="demo_app",
        application_environment_id="env-prod",
        manifest_release_id="rel-1",
        binding_revisions=(1,),
        entries=(entry,),
    ).with_revision_id()
    revision_id = roster.effective_roster_revision_id
    with pytest.raises(TypeError):
        roster.entries[0].merged_config["limits"]["rpm"] = 999
    assert roster.with_revision_id().effective_roster_revision_id == revision_id


def test_canonical_namespace_matches_architecture_doc() -> None:
    repo = Path(__file__).resolve().parents[3]
    architecture = (repo / "docs" / "project" / "architecture" / "AGENT_DISTRIBUTION.md").read_text(
        encoding="utf-8"
    )
    assert "intergrax/agent_distribution/" in architecture
    assert "intergrax/core/agent_distribution/" not in architecture
    assert (repo / "intergrax" / "agent_distribution" / "__init__.py").is_file()


def test_runtime_revision_store_expected_state_is_typed() -> None:
    signature = inspect.signature(RuntimeRevisionStore.persist_candidate_revision)
    annotation = signature.parameters["expected_revision_state"].annotation
    assert annotation in {RuntimeRevisionState | None, "RuntimeRevisionState | None"}


def test_malformed_package_digest_rejected_consistently() -> None:
    with pytest.raises(ValidationError):
        AgentArtifactMetadata(
            package_digest="not-a-digest",
            artifact_store_ref="store://artifacts/1",
            distribution_package_id="intergrax-local-search-agent",
        )
    with pytest.raises(ValidationError):
        EffectiveRosterEntry(
            logical_agent_id="search",
            installation_slot_id="slot-1",
            package_digest="sha256:deadbeef",
            distribution_package_id="intergrax-local-search-agent",
            effective_enablement=True,
        )
    with pytest.raises(ValidationError):
        InstalledAgentPackageRequirement(
            distribution_package_id="intergrax-local-search-agent",
            package_digest="digest-only",
            agent_project_metadata_ref="meta://1",
        )


def test_installation_invalid_lifecycle_combinations_fail() -> None:
    with pytest.raises(ValidationError):
        AgentInstallationRecord(
            installation_id="inst-1",
            installation_slot_id="slot-1",
            environment_id="env-1",
            package_identity=_PACKAGE,
            installation_state=InstallationState.INSTALLED_ACTIVE,
            active_for_slot=False,
            artifact_store_ref="store://artifacts/1",
            trust_record=_trust_record(),
        )


def test_runtime_revision_requires_lock_graph_and_artifact_for_validated() -> None:
    with pytest.raises(ValidationError):
        RuntimeRevision(
            runtime_revision_id="rev-1",
            application_environment_id="env-prod",
            application_release_id="rel-1",
            platform_version="0.1.0",
            effective_roster_revision_id="roster-hash",
            revision_state=RuntimeRevisionState.VALIDATED,
        )


def test_runtime_revision_active_requires_activated_at() -> None:
    with pytest.raises(ValidationError):
        RuntimeRevision(
            runtime_revision_id="rev-1",
            application_environment_id="env-prod",
            application_release_id="rel-1",
            platform_version="0.1.0",
            effective_roster_revision_id="roster-hash",
            materialized_runtime_lock_id="lock-1",
            materialized_runtime_lock_digest="lock-digest",
            runtime_graph_digest="graph-digest",
            materialization_artifact_digest="artifact-digest",
            materialization_topology=MaterializationTopology.VENV_BUNDLE,
            revision_state=RuntimeRevisionState.ACTIVE,
        )


def test_materialized_runtime_lock_deterministic_digest() -> None:
    lock = MaterializedRuntimeLock(
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
        inputs_digest="inputs-1",
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name="requests",
                version="2.32.0",
            ),
        ),
        agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST,
                role="direct",
            ),
        ),
    )
    first = lock.with_content_identity()
    second = lock.with_content_identity()
    assert first.lock_digest == second.lock_digest
    assert first.lock_id == first.lock_digest


def test_effective_roster_revision_is_deterministic() -> None:
    entry = EffectiveRosterEntry(
        logical_agent_id="search",
        installation_slot_id="slot-1",
        package_digest=_DIGEST,
        distribution_package_id="intergrax-local-search-agent",
        effective_enablement=True,
    )
    roster = EffectiveRoster(
        application_id="demo_app",
        application_environment_id="env-prod",
        manifest_release_id="rel-1",
        binding_revisions=(1,),
        entries=(entry,),
    )
    assert roster.compute_revision_id() == roster.with_revision_id().effective_roster_revision_id


def test_dependency_resolver_input_digest_is_stable() -> None:
    spec = CandidateDependencySpecification(
        application_release_id="rel-1",
        platform_version="0.1.0",
        repository_declaration=RepositoryDependencyDeclaration(application_release_id="rel-1"),
        agent_packages=(
            InstalledAgentPackageRequirement(
                distribution_package_id="intergrax-local-search-agent",
                package_digest=_DIGEST,
                agent_project_metadata_ref="meta://1",
            ),
        ),
    )
    resolver_input = DependencyResolverInput(
        specification=spec,
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
    )
    assert resolver_input.inputs_digest() == resolver_input.inputs_digest()


def test_installation_state_enum_vocabulary() -> None:
    assert InstallationState.INSTALLED_ACTIVE.value == "installed_active"
    assert RuntimeRevisionState.ACTIVE.value == "active"


def test_agent_distribution_package_has_no_forbidden_imports() -> None:
    repo = Path(__file__).resolve().parents[3]
    package_root = repo / "intergrax" / "agent_distribution"
    agent_dirs = {
        p.name
        for p in (repo / "agents").iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and not p.name.startswith("_")
    }
    app_dirs = {
        p.name
        for p in (repo / "applications").iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.append(node.module)
            for module in modules:
                top = module.split(".", 1)[0]
                if top == "agents" or top in agent_dirs:
                    violations.append(f"{path.relative_to(repo)} imports {module}")
                if top == "applications" or top in app_dirs:
                    violations.append(f"{path.relative_to(repo)} imports {module}")
    assert not violations, "\n".join(violations)


def test_validated_runtime_revision_accepts_required_identities() -> None:
    revision = RuntimeRevision(
        runtime_revision_id="rev-active",
        application_environment_id="env-prod",
        application_release_id="rel-1",
        platform_version="0.1.0",
        effective_roster_revision_id="roster-hash",
        materialized_runtime_lock_id="lock-1",
        materialized_runtime_lock_digest="lock-digest",
        runtime_graph_digest="graph-digest",
        materialization_artifact_digest="artifact-digest",
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )
    assert revision.revision_state is RuntimeRevisionState.VALIDATED
