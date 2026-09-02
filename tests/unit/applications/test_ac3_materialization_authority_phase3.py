# © Artur Czarnecki. All rights reserved.

"""ADR-AGENT-006 Phase 3 — canonical production registry projection authority."""

from __future__ import annotations

import inspect
import json
import textwrap
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import (
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.in_memory_stores import (
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.applications._shared.production_agent_platform_runtime import (
    AgentPlatformRuntimeStores,
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    ProductionRegistryProjectionInputError,
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
    production_test_artifact_locator,
)
from intergrax.applications._shared.registry_projection import (
    InMemoryRuntimeRegistryProjectionStore,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
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

_FORBIDDEN_PUBLIC_PARAMS = frozenset(
    {
        "runtime_revision",
        "materialized_runtime_lock",
        "artifact_locator",
        "materialization_artifact_digest",
    }
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
        name="AC3 Phase 3",
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
    stores: AgentPlatformRuntimeStores,
    *,
    revision: RuntimeRevision,
    lock: MaterializedRuntimeLock,
    artifact_root: Path,
    digest: str,
) -> RuntimeMaterializationRecord:
    record = _materialization_record(revision, lock, artifact_root, digest)
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(record)
    stores.revision_store.persist_candidate_revision(revision)
    return record


def _canonical_fixture(
    tmp_path: Path,
    *,
    revision_id: str = "rev-prod-a",
    marker: str = "artifact-a",
    stores: AgentPlatformRuntimeStores | None = None,
):
    runtime = build_production_agent_platform_runtime()
    stores = stores or runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path, marker=marker)
    manifest = _manifest()
    roster = _roster()
    revision = _revision(revision_id, roster=roster, artifact_digest=digest, lock=lock)
    record = _seed_canonical_authority(
        stores,
        revision=revision,
        lock=lock,
        artifact_root=artifact_root,
        digest=digest,
    )
    return {
        "stores": stores,
        "manifest": manifest,
        "roster": roster,
        "revision": revision,
        "lock": lock,
        "artifact_root": artifact_root,
        "digest": digest,
        "record": record,
    }


def _build_bundle(fixture: dict) -> object:
    return build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=fixture["revision"].runtime_revision_id,
        effective_roster=fixture["roster"],
        manifest=fixture["manifest"],
        build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
        stores=fixture["stores"],
    )


def test_production_runtime_revision_and_lock_stores_share_distribution_state() -> None:
    runtime = build_production_agent_platform_runtime()
    revision_store = runtime.stores.revision_store
    lock_store = runtime.stores.lock_store
    materialization_store = runtime.stores.materialization_store
    assert isinstance(revision_store, InMemoryRuntimeRevisionStore)
    assert isinstance(lock_store, InMemoryMaterializedRuntimeLockStore)
    assert isinstance(materialization_store, InMemoryRuntimeMaterializationStore)
    assert revision_store.state is runtime.distribution_state
    assert lock_store.state is runtime.distribution_state
    assert materialization_store.state is runtime.distribution_state


def test_public_production_projection_api_has_no_caller_authority_params() -> None:
    for name in (
        "build_production_registry_projection_input_bundle_for_revision",
        "build_production_registry_projection_for_revision",
    ):
        signature = inspect.signature(
            getattr(
                __import__(
                    "intergrax.applications._shared.production_registry_projection_input_bundle",
                    fromlist=[name],
                ),
                name,
            )
        )
        assert _FORBIDDEN_PUBLIC_PARAMS.isdisjoint(signature.parameters)


def test_canonical_projection_uses_store_revision_not_caller_object(
    tmp_path: Path,
) -> None:
    fixture = _canonical_fixture(tmp_path)
    bundle = _build_bundle(fixture)
    assert (
        bundle.runtime_revision.runtime_revision_id
        == fixture["revision"].runtime_revision_id
    )
    assert bundle.runtime_revision is fixture["revision"] or (
        bundle.runtime_revision.runtime_revision_id
        == fixture["revision"].runtime_revision_id
    )


def test_wrong_application_scope_fails_closed(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path)
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="application_id mismatch"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id="app-b",
            application_environment_id=_ENV,
            runtime_revision_id=fixture["revision"].runtime_revision_id,
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_wrong_environment_scope_fails_closed(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path)
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="environment_id mismatch"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id="env-staging",
            runtime_revision_id=fixture["revision"].runtime_revision_id,
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_missing_revision_fails_closed(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path)
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="runtime revision not found"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-missing",
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_missing_materialization_record_fails_closed(tmp_path: Path) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(
        tmp_path, marker="rev-no-materialization"
    )
    manifest = _manifest()
    roster = _roster()
    revision = _revision(
        "rev-no-materialization",
        roster=roster,
        artifact_digest=digest,
        lock=lock,
    )
    stores.lock_store.persist_lock(lock)
    stores.revision_store.persist_candidate_revision(revision)
    with pytest.raises(
        ProductionRegistryProjectionInputError,
        match="missing canonical materialization record",
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def _seed_mismatch_fixture(
    tmp_path: Path,
    *,
    revision_id: str,
    record_updates: dict[str, object],
):
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path, marker=revision_id)
    manifest = _manifest()
    roster = _roster()
    revision = _revision(revision_id, roster=roster, artifact_digest=digest, lock=lock)
    record = _materialization_record(revision, lock, artifact_root, digest).model_copy(
        update=record_updates,
    )
    stores.lock_store.persist_lock(lock)
    stores.materialization_store.persist(record)
    stores.revision_store.persist_candidate_revision(revision)
    return {
        "stores": stores,
        "manifest": manifest,
        "roster": roster,
        "revision": revision,
    }


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    (
        ("application_id", "app-b", "application id mismatch"),
        (
            "application_environment_id",
            "env-staging",
            "application environment id mismatch",
        ),
        (
            "materialization_topology",
            MaterializationTopology.OCI_IMAGE,
            "topology mismatch",
        ),
        (
            "materialization_artifact_digest",
            "sha256:" + ("9" * 64),
            "artifact digest mismatch",
        ),
        ("materialized_runtime_lock_id", "sha256:" + ("f" * 64), "lock id mismatch"),
        (
            "materialized_runtime_lock_digest",
            "sha256:" + ("8" * 64),
            "lock digest mismatch",
        ),
    ),
)
def test_materialization_record_mismatch_fails_closed(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    match: str,
) -> None:
    fixture = _seed_mismatch_fixture(
        tmp_path,
        revision_id=f"rev-mismatch-{field_name}",
        record_updates={field_name: field_value},
    )
    with pytest.raises(ProductionRegistryProjectionInputError, match=match):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=fixture["revision"].runtime_revision_id,
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_missing_lock_fails_closed(tmp_path: Path) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(tmp_path, marker="rev-no-lock")
    manifest = _manifest()
    roster = _roster()
    revision = _revision(
        "rev-no-lock", roster=roster, artifact_digest=digest, lock=lock
    )
    stores.materialization_store.persist(
        _materialization_record(revision, lock, artifact_root, digest)
    )
    stores.revision_store.persist_candidate_revision(revision)
    with pytest.raises(
        ProductionRegistryProjectionInputError,
        match="canonical materialized runtime lock not found",
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def test_lock_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    runtime = build_production_agent_platform_runtime()
    stores = runtime.stores
    artifact_root, digest, lock = _build_echo_artifact(
        tmp_path, marker="rev-lock-digest"
    )
    manifest = _manifest()
    roster = _roster()
    wrong_digest = "sha256:" + ("9" * 64)
    revision = _revision(
        "rev-lock-digest",
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
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="lock digest mismatch"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=revision.runtime_revision_id,
            effective_roster=roster,
            manifest=manifest,
            build_context=ApplicationBuildContext.for_manifest(manifest),
            stores=stores,
        )


def test_locator_from_materialization_store_resolves_real_venv(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path, revision_id="rev-locator-proof")
    projection = build_production_registry_projection_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=fixture["revision"].runtime_revision_id,
        effective_roster=fixture["roster"],
        manifest=fixture["manifest"],
        build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
        stores=fixture["stores"],
    )
    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert projection.evidence.materialization_artifact_digest == fixture["digest"]
    assert fixture["record"].artifact_locator.startswith("test://")


def test_artifact_digest_from_canonical_state_fails_on_tamper(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path, revision_id="rev-digest-tamper")
    tampered = (
        fixture["artifact_root"] / "site-packages" / "example_agent" / "factory.py"
    )
    tampered.write_text("MARKER = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ProductionRegistryProjectionInputError, match="digest mismatch"):
        build_production_registry_projection_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=fixture["revision"].runtime_revision_id,
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_matching_roster_revision_allowed(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path, revision_id="rev-roster-ok")
    bundle = _build_bundle(fixture)
    assert (
        bundle.effective_roster.effective_roster_revision_id
        == fixture["revision"].effective_roster_revision_id
    )


def test_wrong_roster_revision_fails_closed(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path, revision_id="rev-roster-bad")
    roster = fixture["roster"].model_copy(
        update={"effective_roster_revision_id": "sha256:" + ("9" * 64)},
    )
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="effective_roster_revision_id"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=fixture["revision"].runtime_revision_id,
            effective_roster=roster,
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=fixture["stores"],
        )


def test_projection_store_not_used_as_revision_authority(tmp_path: Path) -> None:
    fixture = _canonical_fixture(tmp_path, revision_id="rev-store-boundary")
    stores = fixture["stores"]
    assert isinstance(
        stores.registry_projection_store, InMemoryRuntimeRegistryProjectionStore
    )
    with pytest.raises(
        ProductionRegistryProjectionInputError, match="runtime revision not found"
    ):
        build_production_registry_projection_input_bundle_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id="rev-only-in-projection-store",
            effective_roster=fixture["roster"],
            manifest=fixture["manifest"],
            build_context=ApplicationBuildContext.for_manifest(fixture["manifest"]),
            stores=stores,
        )
