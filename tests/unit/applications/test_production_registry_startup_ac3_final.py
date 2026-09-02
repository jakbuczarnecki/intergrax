# © Artur Czarnecki. All rights reserved.

"""AC-3-FINAL production revision-bound registry startup proofs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.admin_models import ActivateRuntimeRevisionRequest
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import (
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.applications._shared.harness_registry_authority import (
    HarnessHostRegistryAuthorityError,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    create_reference_production_process_composition,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
    production_test_artifact_locator,
    resolve_production_artifact_root,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_APP = "app_a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_DIGEST_A = "sha256:" + ("a" * 64)
_GRAPH_DIGEST = "sha256:" + ("c" * 64)
_FACTORY_REF = AgentBindingFactoryReference(
    factory_path="example_agent.factory.build_agent",
)


def _write_lock_manifest(
    artifact_root: Path,
    *,
    package_digest: str = _DIGEST_A,
) -> MaterializedRuntimeLock:
    lock = MaterializedRuntimeLock(
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
        inputs_digest="inputs-1",
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name="pkg-search",
                version="1.0.0",
                package_digest=package_digest,
            ),
        ),
        agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id="pkg-search",
                package_digest=package_digest,
                role="direct",
            ),
        ),
    ).with_content_identity()
    (artifact_root / RUNTIME_LOCK_MANIFEST_FILENAME).write_text(
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return lock


def _build_echo_artifact(
    tmp_path: Path,
    *,
    marker: str = "artifact-a",
) -> tuple[Path, str, MaterializedRuntimeLock]:
    artifact_root = tmp_path / f"artifact-{marker}"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    package_dir = site_packages / "example_agent"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            f"""
            from echo.echo_agent import EchoAgent

            MARKER = {marker!r}

            def build_agent(ctx, binding):
                return EchoAgent()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    lock = _write_lock_manifest(artifact_root)
    digest = directory_content_digest(artifact_root)
    return artifact_root, digest, lock


def _manifest() -> ApplicationManifest:
    return ApplicationManifest.lab(
        app_id=_APP,
        name="AC3 Final",
        agents=[AgentBinding.mount(EchoAgent, contract_id="search")],
    )


def _roster(
    *,
    roster_revision_seed: str = "roster-a",
) -> EffectiveRoster:
    roster = EffectiveRoster(
        application_id=_APP,
        application_environment_id=_ENV,
        manifest_release_id=_RELEASE,
        entries=(
            EffectiveRosterEntry(
                logical_agent_id="search",
                installation_slot_id="slot-search",
                package_digest=_DIGEST_A,
                distribution_package_id="pkg-search",
                effective_enablement=True,
                factory_reference=_FACTORY_REF,
                manifest_origin_ref="manifest:agents/search",
            ),
        ),
    ).with_revision_id()
    if roster_revision_seed != "roster-a":
        return roster.model_copy(
            update={"effective_roster_revision_id": roster_revision_seed},
        )
    return roster


def _revision(
    revision_id: str,
    *,
    roster: EffectiveRoster,
    artifact_digest: str,
    lock: MaterializedRuntimeLock,
    state: RuntimeRevisionState = RuntimeRevisionState.VALIDATED,
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=roster.effective_roster_revision_id or "",
        installed_agent_package_digests=(_DIGEST_A,),
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=artifact_digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=state,
        activated_at=None,
    )


def _materialization_record(
    revision: RuntimeRevision,
    lock: MaterializedRuntimeLock,
    artifact_root: Path,
    digest: str,
) -> RuntimeMaterializationRecord:
    return RuntimeMaterializationRecord(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        artifact_locator=production_test_artifact_locator(artifact_root),
        materialization_artifact_digest=digest,
        materialized_runtime_lock_id=lock.lock_id,
        materialized_runtime_lock_digest=lock.lock_digest,
    )


def _seed_canonical_authority(
    stores,
    *,
    revision: RuntimeRevision,
    lock: MaterializedRuntimeLock,
    artifact_root: Path,
    digest: str,
) -> str:
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(
        _materialization_record(revision, lock, artifact_root, digest)
    )
    stores.revision_store.persist_candidate_revision(revision)
    return production_test_artifact_locator(artifact_root)


def _canonical_bundle(
    tmp_path: Path,
    *,
    revision_id: str = "rev-prod-a",
    marker: str = "artifact-a",
    stores=None,
):
    runtime = build_production_agent_platform_runtime()
    stores = stores or runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path, marker=marker)
    manifest = _manifest()
    roster = _roster()
    revision = _revision(revision_id, roster=roster, artifact_digest=digest, lock=lock)
    locator = _seed_canonical_authority(
        stores,
        revision=revision,
        lock=lock,
        artifact_root=artifact_root,
        digest=digest,
    )
    bundle = build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=revision_id,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        stores=stores,
    )
    return bundle, locator, artifact_root, digest, lock, stores


def _activation_request(bundle, locator: str) -> ActivateRuntimeRevisionRequest:
    revision_id = bundle.runtime_revision.runtime_revision_id
    return ActivateRuntimeRevisionRequest(
        mutation_id=f"activate:{revision_id}",
        runtime_revision_id=revision_id,
        artifact_locator=locator,
        expected_artifact_digest=bundle.materialization_artifact_digest,
        expected_serving_pointer_revision=0,
        expected_prior_traffic_revision_id=None,
    )


def _activate_bundle(launcher, bundle, locator: str, *, principal) -> str:
    launcher.deploy_and_activate(
        bundle,
        _activation_request(bundle, locator),
        principal=principal,
        admission_mutation_id=reference_admission_mutation_id(
            bundle.runtime_revision.runtime_revision_id,
        ),
    )
    return bundle.runtime_revision.runtime_revision_id


def test_production_bundle_uses_real_venv_resolver(tmp_path: Path) -> None:
    bundle, _, _, digest, _, stores = _canonical_bundle(tmp_path)
    projection = build_production_registry_projection_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=bundle.runtime_revision.runtime_revision_id,
        effective_roster=bundle.effective_roster,
        manifest=bundle.manifest,
        build_context=bundle.build_context,
        stores=stores,
    )
    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert projection.evidence.materialization_artifact_digest == digest


def test_production_bundle_rejects_reference_locator(tmp_path: Path) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path)
    manifest = _manifest()
    roster = _roster()
    revision = _revision(
        "rev-ref-locator", roster=roster, artifact_digest=digest, lock=lock
    )
    bad_record = _materialization_record(
        revision, lock, artifact_root, digest
    ).model_copy(
        update={"artifact_locator": "reference://process-local/venv-bundle/rev-1"},
    )
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(bad_record)
    stores.revision_store.persist_candidate_revision(revision)
    with pytest.raises(ProductionRegistryProjectionInputError, match="reference://"):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def test_production_bundle_roster_mismatch_fails_closed(tmp_path: Path) -> None:
    bundle, _, _, _, _, stores = _canonical_bundle(tmp_path)
    mismatched = bundle.effective_roster.model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("9" * 64)},
    )
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="effective_roster_revision_id"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=bundle.runtime_revision.runtime_revision_id,
            effective_roster=mismatched,
            manifest=bundle.manifest,
            build_context=bundle.build_context,
            stores=stores,
        )


def test_production_bundle_lock_mismatch_fails_closed(tmp_path: Path) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path)
    manifest = _manifest()
    roster = _roster()
    wrong_digest = "sha256:" + ("9" * 64)
    revision = _revision(
        "rev-lock-mismatch",
        roster=roster,
        artifact_digest=digest,
        lock=lock,
    ).model_copy(update={"materialized_runtime_lock_digest": wrong_digest})
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(
        _materialization_record(revision, lock, artifact_root, digest).model_copy(
            update={"materialized_runtime_lock_digest": wrong_digest},
        )
    )
    stores.revision_store.persist_candidate_revision(revision)
    with pytest.raises(ProductionRegistryProjectionInputError, match="lock digest"):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def test_no_active_revision_fails_closed() -> None:
    composition = create_reference_production_process_composition()
    with pytest.raises(
        HarnessHostRegistryAuthorityError, match="no active traffic-serving"
    ):
        bootstrap_production_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            stores=composition.agent_platform_runtime.stores,
        )


def test_reference_manifest_bundle_cannot_rescue_production_artifact_mismatch(
    tmp_path: Path,
) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path)
    manifest = _manifest()
    roster = _roster()
    revision = _revision(
        "rev-prod-only", roster=roster, artifact_digest=digest, lock=lock
    )
    tampered_revision = revision.model_copy(
        update={"materialization_artifact_digest": "sha256:" + ("9" * 64)},
    )
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(
        _materialization_record(revision, lock, artifact_root, digest)
    )
    stores.revision_store.persist_candidate_revision(tampered_revision)
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="artifact digest mismatch"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=tampered_revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def test_production_e2e_activate_and_serve_real_artifact(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    stores = composition.agent_platform_runtime.stores
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    bundle, locator, _, digest, _, _ = _canonical_bundle(
        tmp_path,
        revision_id="rev-e2e",
        stores=stores,
    )
    revision_id = _activate_bundle(
        launcher,
        bundle,
        locator,
        principal=governance.principal,
    )
    resolved = bootstrap_production_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == revision_id
    assert resolved.evidence.materialization_artifact_digest == digest
    assert resolved.agent_registry.list_agent_ids() == ["search"]
    assert len(resolved.agent_registry.find_by_capability("echo.basic")) == 1


def test_ready_n_plus_one_does_not_switch_serving(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    stores = composition.agent_platform_runtime.stores
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    bundle_n, locator_n, _, _, _, _ = _canonical_bundle(
        tmp_path,
        revision_id="rev-n",
        marker="n",
        stores=stores,
    )
    _activate_bundle(launcher, bundle_n, locator_n, principal=governance.principal)

    bundle_n1, locator_n1, _, _, _, _ = _canonical_bundle(
        tmp_path / "n1",
        revision_id="rev-n1",
        marker="n1",
        stores=stores,
    )
    launcher.services.projection_input_store.register(bundle_n1)
    launcher.services.revision_service.persist_candidate_revision(
        bundle_n1.runtime_revision.model_copy(
            update={"revision_state": RuntimeRevisionState.CANDIDATE},
        ),
    )
    launcher.services.revision_service.mark_validated(
        "rev-n1",
        validated_revision=bundle_n1.runtime_revision,
    )
    launcher.services.activation_service.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        artifact_locator=locator_n1,
    )

    resolved = bootstrap_production_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == "rev-n"


def test_commit_n_plus_one_switches_serving(tmp_path: Path) -> None:
    composition = create_reference_production_process_composition()
    stores = composition.agent_platform_runtime.stores
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    bundle_n, locator_n, _, _, _, _ = _canonical_bundle(
        tmp_path,
        revision_id="rev-n",
        marker="n",
        stores=stores,
    )
    _activate_bundle(launcher, bundle_n, locator_n, principal=governance.principal)

    bundle_n1, locator_n1, _, digest_n1, _, _ = _canonical_bundle(
        tmp_path / "n1",
        revision_id="rev-n1",
        marker="n1",
        stores=stores,
    )
    launcher.services.projection_input_store.register(bundle_n1)
    launcher.services.revision_service.persist_candidate_revision(
        bundle_n1.runtime_revision.model_copy(
            update={"revision_state": RuntimeRevisionState.CANDIDATE},
        ),
    )
    launcher.services.revision_service.mark_validated(
        "rev-n1",
        validated_revision=bundle_n1.runtime_revision,
    )
    launcher.services.activation_service.prepare_candidate(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        artifact_locator=locator_n1,
    )
    launcher.services.activation_service.commit_activation(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id="rev-n1",
        expected_prior_traffic_revision_id="rev-n",
        expected_serving_pointer_revision=1,
        expected_artifact_digest=digest_n1,
    )

    resolved = bootstrap_production_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        stores=composition.agent_platform_runtime.stores,
    )
    assert resolved.evidence.runtime_revision_id == "rev-n1"
    assert resolved.evidence.materialization_artifact_digest == digest_n1


def test_same_process_composition_lifecycle_and_serving_share_stores(
    tmp_path: Path,
) -> None:
    composition = create_reference_production_process_composition()
    stores = composition.agent_platform_runtime.stores
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_ENV)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    bundle, locator, _, _, _, _ = _canonical_bundle(
        tmp_path,
        revision_id="rev-shared",
        stores=stores,
    )
    _activate_bundle(launcher, bundle, locator, principal=governance.principal)
    serving = stores.serving_store.get_serving_record(_APP, _ENV)
    assert serving is not None
    assert serving.traffic_serving_revision_id == "rev-shared"
    projection = stores.registry_projection_store.get("rev-shared")
    assert projection is not None
    resolved = bootstrap_production_registry_projection(
        application_id=_APP,
        application_environment_id=_ENV,
        stores=stores,
    )
    assert resolved.evidence.runtime_revision_id == serving.traffic_serving_revision_id


def test_resolve_production_artifact_root_round_trip(tmp_path: Path) -> None:
    artifact_root, _, _ = _build_echo_artifact(tmp_path)
    locator = production_test_artifact_locator(artifact_root)
    assert resolve_production_artifact_root(locator) == artifact_root.resolve()
