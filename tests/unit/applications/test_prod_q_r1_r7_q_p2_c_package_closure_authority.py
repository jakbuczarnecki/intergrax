# © Artur Czarnecki. All rights reserved.

"""P2-C-ID-R1 — revision-bound package closure contract authority."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.applications._shared.package_wiring import (
    assert_manifest_package_closure,
    collect_application_dependencies,
    resolve_strict_package_contract_authority_for_serving_handoff,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection import build_registry_projection
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications._shared.roster_agent_contract_authority import (
    StrictAgentContractAuthorityError,
)
from intergrax.applications._shared.serving_runtime_handoff import (
    ServingRuntimeHandoff,
    ServingUnitIdentity,
)
from intergrax.applications._shared.strict_product_manifest_ci_gates import (
    manifest_ci_contract_authority,
)
from intergrax.applications.contracts.application_package import ApplicationDependencyKind
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from echo.echo_agent import EchoAgent
from testing_support.agent_distribution.install_contract_authority_fixtures import (
    package_contract_authority_record,
)
from tests.unit.applications.test_production_registry_startup_ac3_final import (
    _APP,
    _DIGEST_A,
    _ENV,
    _ac3_revision_bound_host_manifest,
    _authority,
    _build_echo_artifact,
    _materialization_record,
    _revision,
    _roster,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _search_contract():
    sys.path.insert(0, "agents/echo")
    try:
        from contract import build_agent_contract

        return build_agent_contract().model_copy(update={"id": "search", "name": "search"})
    finally:
        sys.path.pop(0)


def _seed_declarative_revision_bundle(tmp_path: Path):
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path)
    manifest = _ac3_revision_bound_host_manifest()
    roster = _roster()
    revision = _revision("rev-p2c-declarative", roster=roster, artifact_digest=digest, lock=lock)
    contract = _search_contract()
    stores.artifact_metadata_store.persist_package_contract_authority(
        package_contract_authority_record(
            contract=contract,
            package_digest=_DIGEST_A,
            distribution_package_id="pkg-search",
            artifact_store_ref="store://artifacts/slot-search",
            agent_project_metadata_ref="meta://search",
        ),
    )
    stores.effective_roster_snapshot_store.persist(roster)
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(
        _materialization_record(revision, lock, artifact_root, digest),
    )
    stores.revision_store.persist_candidate_revision(revision)
    bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=revision.runtime_revision_id,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        authority=_authority(stores),
    )
    projection = build_registry_projection(bundle)
    return runtime, bundle, projection, manifest, revision


def test_declarative_search_emits_version_pinned_agent_dependency(tmp_path: Path) -> None:
    _runtime, bundle, _projection, manifest, _revision = _seed_declarative_revision_bundle(
        tmp_path,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    deps = collect_application_dependencies(
        manifest,
        env,
        contract_authority=bundle.roster_contract_authority,
    )
    agent_deps = [dep for dep in deps if dep.kind is ApplicationDependencyKind.AGENT]
    assert len(agent_deps) == 1
    assert agent_deps[0].ref == "search"
    assert agent_deps[0].version_constraint.startswith("=")
    assert agent_deps[0].version_constraint != "*"


def test_missing_authority_fail_closed_for_declarative_search() -> None:
    manifest = _ac3_revision_bound_host_manifest()
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    with pytest.raises(StrictAgentContractAuthorityError):
        collect_application_dependencies(manifest, env, contract_authority=None)


def test_wrong_contract_id_fail_closed(tmp_path: Path) -> None:
    _runtime, bundle, _projection, manifest, _revision = _seed_declarative_revision_bundle(
        tmp_path,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    bad_manifest = manifest.model_copy(
        update={"agents": [AgentBinding(contract_id="missing-agent")]},
    )
    with pytest.raises(StrictAgentContractAuthorityError):
        collect_application_dependencies(
            bad_manifest,
            env,
            contract_authority=bundle.roster_contract_authority,
        )


def test_wrong_environment_rejected_before_package_validation(tmp_path: Path) -> None:
    platform, bundle, projection, manifest, revision = _seed_declarative_revision_bundle(
        tmp_path,
    )
    handoff = ServingRuntimeHandoff(
        identity=ServingUnitIdentity(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            serving_unit_ref="unit-1",
            materialization_artifact_digest=revision.materialization_artifact_digest or "",
            materialization_topology=MaterializationTopology.VENV_BUNDLE,
        ),
        runtime_agent_factory_resolver=bundle.factory_resolver,
    )
    wrong_env = ApplicationEnvironmentProfile.product_defaults(profile_id="wrong-env")
    with pytest.raises(StrictAgentContractAuthorityError, match="profile_id disagrees"):
        resolve_strict_package_contract_authority_for_serving_handoff(
            manifest=manifest,
            env=wrong_env,
            handoff=handoff,
            registry_projection=projection,
            registry_projection_authority=platform.registry_projection_authority,
        )


def test_canonical_package_validator_still_active(tmp_path: Path) -> None:
    _platform, bundle, projection, manifest, _revision = _seed_declarative_revision_bundle(
        tmp_path,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    snapshot = HarnessRegistrySnapshot(
        integration_profile=IntegrationProfile(),
        tool_registry=None,
        skill_registry=None,
        agent_registry=projection.agent_registry,
        prompt_registry=None,
        policy_bundle=None,
    )
    assert_manifest_package_closure(
        manifest,
        env,
        snapshot,
        contract_authority=bundle.roster_contract_authority,
    )


def test_lab_ci_manifest_authority_supports_echo_mount() -> None:
    manifest = ApplicationManifest.lab(
        app_id="lab_ci",
        name="Lab CI",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo")],
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="lab_ci.scaffold")
    authority = manifest_ci_contract_authority(manifest)
    deps = collect_application_dependencies(manifest, env, contract_authority=authority)
    assert any(dep.kind is ApplicationDependencyKind.AGENT for dep in deps)


def test_gca_factory_source_has_no_lab_compat_helper() -> None:
    source = Path("applications/governed_contractor_application/host/factory.py").read_text(
        encoding="utf-8",
    )
    assert "materialize_manifest_contract_authority_lab_compat" not in source
