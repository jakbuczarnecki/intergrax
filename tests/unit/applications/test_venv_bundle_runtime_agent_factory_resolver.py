# © Artur Czarnecki. All rights reserved.

"""Production VENV_BUNDLE RuntimeAgentFactoryResolver tests."""

from __future__ import annotations

import json
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from echo.echo_agent import EchoAgent
from intergrax.agent_distribution.binding import AgentBindingFactoryReference
from intergrax.agent_distribution.dependency import (
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_LOCK_MANIFEST_FILENAME,
    directory_content_digest,
)
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.roster import EffectiveRoster, EffectiveRosterEntry
from intergrax.applications._shared.registry_projection import (
    RegistryProjectionInputBundle,
    build_registry_projection,
)
from intergrax.applications._shared.runtime_agent_factory_resolver import (
    RuntimeAgentFactoryResolutionError,
)
from intergrax.applications._shared.venv_bundle_runtime_agent_factory_resolver import (
    VenvBundleRuntimeAgentFactoryResolver,
    build_production_runtime_agent_factory_resolver,
    is_production_runtime_factory_adapter_deferred,
    production_runtime_factory_topology_status,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_LOCK_ID = "lock-venv-1"
_LOCK_DIGEST = "sha256:" + ("c" * 64)
_GRAPH_DIGEST = "sha256:" + ("d" * 64)
_ARTIFACT_N = "sha256:" + ("e" * 64)
_ARTIFACT_N1 = "sha256:" + ("f" * 64)
_APP = "app_a"
_ENV = "env-prod"
_RELEASE = "rel-1"
_ROSTER = "sha256:" + ("1" * 64)
_FACTORY_PATH = "artifact_agent.factory.build_agent"
_FACTORY_REF = AgentBindingFactoryReference(factory_path=_FACTORY_PATH)


def _write_lock_manifest(
    artifact_root: Path,
    *,
    package_digest: str = _DIGEST_A,
    distribution_package_id: str = "pkg-search",
) -> MaterializedRuntimeLock:
    lock = MaterializedRuntimeLock(
        resolver_algorithm_id="intergrax.test",
        resolver_algorithm_version="1",
        inputs_digest="inputs-1",
        intergrax_version="0.1.0",
        python_version="3.12",
        packages=(
            MaterializedLockPackage(
                distribution_name=distribution_package_id,
                version="1.0.0",
                package_digest=package_digest,
            ),
        ),
        agent_closure=(
            MaterializedAgentClosureEntry(
                distribution_package_id=distribution_package_id,
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


def _write_factory_module(
    site_packages: Path,
    *,
    marker: str,
    module_name: str = "artifact_agent",
) -> None:
    package_dir = site_packages / module_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            f"""
            MARKER = {marker!r}

            def build_agent(ctx, binding):
                return ("factory", MARKER, binding.contract_id)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build_artifact(
    tmp_path: Path,
    *,
    marker: str = "artifact",
    package_digest: str = _DIGEST_A,
    module_name: str = "artifact_agent",
) -> tuple[Path, str, MaterializedRuntimeLock]:
    artifact_root = tmp_path / "artifact"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    _write_factory_module(site_packages, marker=marker, module_name=module_name)
    lock = _write_lock_manifest(artifact_root, package_digest=package_digest)
    digest = directory_content_digest(artifact_root)
    return artifact_root, digest, lock


def _revision(
    revision_id: str,
    *,
    artifact_digest: str,
    lock_id: str,
    lock_digest: str,
    package_digests: tuple[str, ...] = (_DIGEST_A,),
) -> RuntimeRevision:
    return RuntimeRevision(
        runtime_revision_id=revision_id,
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=_ROSTER,
        installed_agent_package_digests=package_digests,
        materialized_runtime_lock_id=lock_id,
        materialized_runtime_lock_digest=lock_digest,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=artifact_digest,
        materialization_topology=MaterializationTopology.VENV_BUNDLE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )


def _resolver(
    artifact_root: Path,
    artifact_digest: str,
) -> VenvBundleRuntimeAgentFactoryResolver:
    return VenvBundleRuntimeAgentFactoryResolver(
        artifact_root=artifact_root,
        expected_artifact_digest=artifact_digest,
    )


def test_topology_status_venv_implemented_oci_deferred() -> None:
    assert production_runtime_factory_topology_status(MaterializationTopology.VENV_BUNDLE) == "implemented"
    assert production_runtime_factory_topology_status(MaterializationTopology.OCI_IMAGE) == "deferred"
    assert (
        production_runtime_factory_topology_status(MaterializationTopology.SANDBOX_SIDECAR)
        == "deferred"
    )
    assert is_production_runtime_factory_adapter_deferred(MaterializationTopology.VENV_BUNDLE) is False
    assert is_production_runtime_factory_adapter_deferred(MaterializationTopology.OCI_IMAGE) is True


def test_resolve_factory_from_artifact(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    resolver = _resolver(artifact_root, digest)
    factory = resolver.resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    result = factory(None, AgentBinding(contract_id="search"))
    assert result == ("factory", "artifact", "search")


def test_artifact_beats_workspace_shadow_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_root = tmp_path / "shadow"
    shadow_site = shadow_root / "site-packages"
    shadow_site.mkdir(parents=True)
    _write_factory_module(shadow_site, marker="shadow", module_name="artifact_agent")
    monkeypatch.setattr(sys, "path", [str(shadow_site), *sys.path])
    artifact_root, digest, lock = _build_artifact(tmp_path / "bundle", marker="artifact")
    revision = _revision("rev-shadow", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    factory = _resolver(artifact_root, digest).resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    assert factory(None, AgentBinding(contract_id="x"))[1] == "artifact"


def test_digest_mismatch_fails_before_load(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    wrong_digest = "sha256:" + ("9" * 64)
    resolver = _resolver(artifact_root, wrong_digest)
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="digest mismatch"):
        resolver.resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )


def test_wrong_package_digest_fails_revision_trust(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision(
        "rev-1",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
        package_digests=(_DIGEST_A,),
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="not part of runtime revision"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_B,
            factory_reference=_FACTORY_REF,
        )


def test_wrong_package_digest_fails_lock_authority(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path, package_digest=_DIGEST_A)
    revision = _revision(
        "rev-1",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
        package_digests=(_DIGEST_A, _DIGEST_B),
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="not authorized by artifact lock"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_B,
            factory_reference=_FACTORY_REF,
        )


def test_wrong_factory_path_fails(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="missing from module"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=AgentBindingFactoryReference(
                factory_path="artifact_agent.factory.missing"
            ),
        )


def test_non_callable_factory_fails(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    package_dir = site_packages / "artifact_agent"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text("build_agent = 1\n", encoding="utf-8")
    lock = _write_lock_manifest(artifact_root)
    digest = directory_content_digest(artifact_root)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="not callable"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )


def test_builder_key_fails_closed(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="builder_key"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=AgentBindingFactoryReference(builder_key="echo"),
        )


def test_multi_revision_module_isolation(tmp_path: Path) -> None:
    artifact_n, digest_n, lock_n = _build_artifact(tmp_path / "n", marker="impl-a")
    artifact_n1, digest_n1, lock_n1 = _build_artifact(
        tmp_path / "n1",
        marker="impl-b",
        package_digest=_DIGEST_B,
    )
    revision_n = _revision(
        "rev-n",
        artifact_digest=digest_n,
        lock_id=lock_n.lock_id or "",
        lock_digest=lock_n.lock_digest or "",
    )
    revision_n1 = _revision(
        "rev-n1",
        artifact_digest=digest_n1,
        lock_id=lock_n1.lock_id or "",
        lock_digest=lock_n1.lock_digest or "",
        package_digests=(_DIGEST_B,),
    )
    resolver_n = _resolver(artifact_n, digest_n)
    resolver_n1 = _resolver(artifact_n1, digest_n1)
    factory_n = resolver_n.resolve_factory(
        runtime_revision=revision_n,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    factory_n1 = resolver_n1.resolve_factory(
        runtime_revision=revision_n1,
        package_digest=_DIGEST_B,
        factory_reference=_FACTORY_REF,
    )
    assert factory_n(None, AgentBinding(contract_id="a"))[1] == "impl-a"
    assert factory_n1(None, AgentBinding(contract_id="b"))[1] == "impl-b"


def test_concurrent_resolution_is_consistent(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path)
    revision = _revision("rev-1", artifact_digest=digest, lock_id=lock.lock_id or "", lock_digest=lock.lock_digest or "")
    resolver = _resolver(artifact_root, digest)

    def _resolve() -> object:
        factory = resolver.resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )
        return factory(None, AgentBinding(contract_id="search"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _resolve(), range(16)))
    assert all(result == ("factory", "artifact", "search") for result in results)


def test_build_production_resolver_rejects_oci() -> None:
    revision = RuntimeRevision(
        runtime_revision_id="rev-oci",
        application_id=_APP,
        application_environment_id=_ENV,
        application_release_id=_RELEASE,
        platform_version="0.1.0",
        effective_roster_revision_id=_ROSTER,
        installed_agent_package_digests=(_DIGEST_A,),
        materialized_runtime_lock_id=_LOCK_ID,
        materialized_runtime_lock_digest=_LOCK_DIGEST,
        runtime_graph_digest=_GRAPH_DIGEST,
        materialization_artifact_digest=_ARTIFACT_N,
        materialization_topology=MaterializationTopology.OCI_IMAGE,
        revision_state=RuntimeRevisionState.VALIDATED,
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="deferred"):
        build_production_runtime_agent_factory_resolver(
            runtime_revision=revision,
            artifact_root=Path("/tmp/unused"),
        )


def _echo_artifact(tmp_path: Path) -> tuple[Path, str, MaterializedRuntimeLock]:
    artifact_root = tmp_path / "echo-artifact"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    echo_src = Path(__file__).resolve().parents[3] / "agents" / "echo"
    echo_dest = site_packages / "echo"
    echo_dest.mkdir()
    for name in ("__init__.py", "echo_agent.py"):
        (echo_dest / name).write_text(
            (echo_src / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    package_dir = site_packages / "artifact_agent"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            """
            from echo.echo_agent import EchoAgent

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


def test_ap10_registry_projection_with_venv_resolver(tmp_path: Path) -> None:
    artifact_root, digest, lock = _echo_artifact(tmp_path)
    manifest = ApplicationManifest.lab(
        app_id=_APP,
        name="App A",
        agents=[AgentBinding.mount(EchoAgent, contract_id="search")],
    )
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
    revision = _revision(
        "rev-ap10",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    revision = revision.model_copy(
        update={"effective_roster_revision_id": roster.effective_roster_revision_id or _ROSTER}
    )
    resolver = build_production_runtime_agent_factory_resolver(
        runtime_revision=revision,
        artifact_root=artifact_root,
        expected_artifact_digest=digest,
    )
    bundle = RegistryProjectionInputBundle(
        runtime_revision=revision,
        effective_roster=roster,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        factory_resolver=resolver,
        builders=None,
        materialization_artifact_digest=digest,
    )
    projection = build_registry_projection(bundle)
    assert projection.agent_registry.list_agent_ids() == ["search"]
    assert projection.evidence.materialization_artifact_digest == digest
    matches = projection.agent_registry.find_by_capability("echo.basic")
    assert len(matches) == 1


def test_production_resolver_has_no_builder_map_import() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "applications"
        / "_shared"
        / "venv_bundle_runtime_agent_factory_resolver.py"
    )
    text = source.read_text(encoding="utf-8")
    for token in ("from intergrax.applications._shared.wiring import", "testing_support"):
        assert token not in text
    assert "InMemoryRuntimeAgentFactoryResolver" not in text
