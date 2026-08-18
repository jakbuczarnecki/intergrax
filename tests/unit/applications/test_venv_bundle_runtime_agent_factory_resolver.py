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
_FACTORY_PATH = "example_agent.factory.build_agent"
_FACTORY_REF = AgentBindingFactoryReference(factory_path=_FACTORY_PATH)
_LEGACY_FACTORY_PATH = "artifact_agent.factory.build_agent"
_LEGACY_FACTORY_REF = AgentBindingFactoryReference(factory_path=_LEGACY_FACTORY_PATH)


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


def _artifact_scope_prefix(artifact_digest: str) -> str:
    normalized = artifact_digest.removeprefix("sha256:").replace("-", "").lower()
    return f"_intergrax_artifact_{normalized}"


def _write_example_agent_package(
    site_packages: Path,
    *,
    marker: str,
    config_marker: str | None = None,
    import_style: str = "relative",
    module_name: str = "example_agent",
) -> None:
    cfg = config_marker if config_marker is not None else marker
    package_dir = site_packages / module_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "config.py").write_text(
        f"MARKER = {cfg!r}\n",
        encoding="utf-8",
        newline="\n",
    )
    (package_dir / "helper.py").write_text(
        (
            "from .config import MARKER as HELPER_MARKER\n\n"
            "def combined():\n"
            "    return HELPER_MARKER\n"
        )
        if import_style == "relative"
        else (
            f"from {module_name}.config import MARKER as HELPER_MARKER\n\n"
            "def combined():\n"
            "    return HELPER_MARKER\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    if import_style == "relative":
        factory_imports = (
            "from .config import MARKER as CONFIG_MARKER\n"
            "from .helper import combined\n"
        )
    else:
        factory_imports = (
            f"from {module_name}.config import MARKER as CONFIG_MARKER\n"
            f"from {module_name}.helper import combined\n"
        )
    (package_dir / "factory.py").write_text(
        (
            f"{factory_imports}\n\n"
            f"MARKER = {marker!r}\n\n"
            "def build_agent(ctx, binding):\n"
            f'    return ("factory", MARKER, CONFIG_MARKER, combined(), binding.contract_id)\n'
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_factory_module(
    site_packages: Path,
    *,
    marker: str,
    module_name: str = "artifact_agent",
    import_style: str = "relative",
) -> None:
    if module_name == "example_agent":
        _write_example_agent_package(
            site_packages,
            marker=marker,
            import_style=import_style,
        )
        return
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
    module_name: str = "example_agent",
    import_style: str = "relative",
    config_marker: str | None = None,
) -> tuple[Path, str, MaterializedRuntimeLock]:
    artifact_root = tmp_path / "artifact"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    _write_factory_module(
        site_packages,
        marker=marker,
        module_name=module_name,
        import_style=import_style,
    )
    if module_name == "example_agent" and config_marker is not None:
        (site_packages / module_name / "config.py").write_text(
            f"MARKER = {config_marker!r}\n",
            encoding="utf-8",
            newline="\n",
        )
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
    assert result == ("factory", "artifact", "artifact", "artifact", "search")


def test_artifact_beats_workspace_shadow_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_root = tmp_path / "shadow"
    shadow_site = shadow_root / "site-packages"
    shadow_site.mkdir(parents=True)
    _write_factory_module(shadow_site, marker="shadow", module_name="example_agent")
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
                factory_path="example_agent.factory.missing"
            ),
        )


def test_non_callable_factory_fails(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    package_dir = site_packages / "example_agent"
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
    assert all(
        result == ("factory", "artifact", "artifact", "artifact", "search") for result in results
    )


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
    package_dir = site_packages / "example_agent"
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


_ECHO_FACTORY_REF = AgentBindingFactoryReference(
    factory_path="example_agent.factory.build_agent"
)


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
                factory_reference=_ECHO_FACTORY_REF,
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
    assert "sys.path.insert" not in text


def test_relative_imports_resolve_inside_artifact(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(
        tmp_path,
        marker="relative-root",
        config_marker="relative-config",
        import_style="relative",
    )
    revision = _revision(
        "rev-relative",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    factory = _resolver(artifact_root, digest).resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    assert factory(None, AgentBinding(contract_id="r")) == (
        "factory",
        "relative-root",
        "relative-config",
        "relative-config",
        "r",
    )


def test_absolute_intra_package_imports_resolve_inside_artifact(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(
        tmp_path,
        marker="absolute-root",
        config_marker="absolute-config",
        import_style="absolute",
    )
    revision = _revision(
        "rev-absolute",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    factory = _resolver(artifact_root, digest).resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    assert factory(None, AgentBinding(contract_id="a")) == (
        "factory",
        "absolute-root",
        "absolute-config",
        "absolute-config",
        "a",
    )


def _invoke_marker(factory: object, contract_id: str) -> str:
    result = factory(None, AgentBinding(contract_id=contract_id))
    return result[1]


def test_nn1_transitive_import_isolation(tmp_path: Path) -> None:
    artifact_n, digest_n, lock_n = _build_artifact(
        tmp_path / "n",
        marker="N",
        config_marker="N-config",
    )
    artifact_n1, digest_n1, lock_n1 = _build_artifact(
        tmp_path / "n1",
        marker="N+1",
        config_marker="N+1-config",
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
    factory_n = _resolver(artifact_n, digest_n).resolve_factory(
        runtime_revision=revision_n,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    factory_n1 = _resolver(artifact_n1, digest_n1).resolve_factory(
        runtime_revision=revision_n1,
        package_digest=_DIGEST_B,
        factory_reference=_FACTORY_REF,
    )
    assert _invoke_marker(factory_n, "1") == "N"
    assert _invoke_marker(factory_n1, "2") == "N+1"
    assert _invoke_marker(factory_n, "3") == "N"


def test_nn1_reverse_load_order(tmp_path: Path) -> None:
    artifact_n, digest_n, lock_n = _build_artifact(tmp_path / "n", marker="N-rev")
    artifact_n1, digest_n1, lock_n1 = _build_artifact(
        tmp_path / "n1",
        marker="N+1-rev",
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
    factory_n1 = _resolver(artifact_n1, digest_n1).resolve_factory(
        runtime_revision=revision_n1,
        package_digest=_DIGEST_B,
        factory_reference=_FACTORY_REF,
    )
    factory_n = _resolver(artifact_n, digest_n).resolve_factory(
        runtime_revision=revision_n,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    assert _invoke_marker(factory_n1, "b") == "N+1-rev"
    assert _invoke_marker(factory_n, "a") == "N-rev"
    assert _invoke_marker(factory_n1, "c") == "N+1-rev"


def test_host_config_shadow_does_not_contaminate(tmp_path: Path) -> None:
    import types

    host_module = types.ModuleType("example_agent.config")
    host_module.MARKER = "HOST"
    sys.modules["example_agent.config"] = host_module
    try:
        artifact_root, digest, lock = _build_artifact(
            tmp_path,
            marker="ARTIFACT",
            config_marker="ARTIFACT",
        )
        revision = _revision(
            "rev-host-shadow",
            artifact_digest=digest,
            lock_id=lock.lock_id or "",
            lock_digest=lock.lock_digest or "",
        )
        factory = _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )
        assert factory(None, AgentBinding(contract_id="x"))[2] == "ARTIFACT"
        assert sys.modules["example_agent.config"].MARKER == "HOST"
    finally:
        sys.modules.pop("example_agent.config", None)


def test_third_party_dependency_shadow_uses_artifact_copy(tmp_path: Path) -> None:
    host_root = tmp_path / "host-shared-dep"
    host_site = host_root / "site-packages"
    host_site.mkdir(parents=True)
    host_pkg = host_site / "shared_dep"
    host_pkg.mkdir()
    (host_pkg / "__init__.py").write_text('VALUE = "HOST"\n', encoding="utf-8")
    sys.path.insert(0, str(host_site))
    try:
        artifact_root = tmp_path / "artifact-dep"
        site_packages = artifact_root / "site-packages"
        site_packages.mkdir(parents=True)
        artifact_pkg = site_packages / "shared_dep"
        artifact_pkg.mkdir()
        (artifact_pkg / "__init__.py").write_text('VALUE = "ARTIFACT"\n', encoding="utf-8")
        agent_pkg = site_packages / "example_agent"
        agent_pkg.mkdir()
        (agent_pkg / "__init__.py").write_text("", encoding="utf-8")
        (agent_pkg / "factory.py").write_text(
            textwrap.dedent(
                """
                import shared_dep

                def build_agent(ctx, binding):
                    return shared_dep.VALUE
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        lock = _write_lock_manifest(artifact_root)
        digest = directory_content_digest(artifact_root)
        revision = _revision(
            "rev-dep",
            artifact_digest=digest,
            lock_id=lock.lock_id or "",
            lock_digest=lock.lock_digest or "",
        )
        factory = _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )
        assert factory(None, AgentBinding(contract_id="dep")) == "ARTIFACT"
    finally:
        sys.path.remove(str(host_site))


def test_failed_import_cleanup_does_not_poison_retry(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact-broken"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    package_dir = site_packages / "example_agent"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "config.py").write_text("MARKER = 'ok'\n", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            """
            from .missing import BROKEN

            def build_agent(ctx, binding):
                return BROKEN
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    lock = _write_lock_manifest(artifact_root)
    digest = directory_content_digest(artifact_root)
    revision = _revision(
        "rev-broken",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    resolver = _resolver(artifact_root, digest)
    with pytest.raises(RuntimeAgentFactoryResolutionError):
        resolver.resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )
    scope_prefix = _artifact_scope_prefix(digest)
    assert not any(name.startswith(scope_prefix) for name in sys.modules)
    (package_dir / "missing.py").write_text("BROKEN = 'fixed'\n", encoding="utf-8")
    digest = directory_content_digest(artifact_root)
    revision = _revision(
        "rev-fixed",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    factory = VenvBundleRuntimeAgentFactoryResolver(
        artifact_root=artifact_root,
        expected_artifact_digest=digest,
    ).resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    assert factory(None, AgentBinding(contract_id="fix")) == "fixed"


def test_missing_artifact_module_fails_without_workspace_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_site = workspace_root / "site-packages"
    workspace_site.mkdir(parents=True)
    ws_pkg = workspace_site / "example_agent"
    ws_pkg.mkdir()
    (ws_pkg / "__init__.py").write_text("", encoding="utf-8")
    (ws_pkg / "missing.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(workspace_site), *sys.path])
    artifact_root = tmp_path / "artifact-missing"
    site_packages = artifact_root / "site-packages"
    site_packages.mkdir(parents=True)
    package_dir = site_packages / "example_agent"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "factory.py").write_text(
        textwrap.dedent(
            """
            from example_agent.missing import X

            def build_agent(ctx, binding):
                return X
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    lock = _write_lock_manifest(artifact_root)
    digest = directory_content_digest(artifact_root)
    revision = _revision(
        "rev-missing",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    with pytest.raises(RuntimeAgentFactoryResolutionError, match="failed to import artifact module"):
        _resolver(artifact_root, digest).resolve_factory(
            runtime_revision=revision,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )


def test_no_canonical_example_agent_modules_cached(tmp_path: Path) -> None:
    artifact_root, digest, lock = _build_artifact(tmp_path, marker="scoped-only")
    revision = _revision(
        "rev-scoped",
        artifact_digest=digest,
        lock_id=lock.lock_id or "",
        lock_digest=lock.lock_digest or "",
    )
    _resolver(artifact_root, digest).resolve_factory(
        runtime_revision=revision,
        package_digest=_DIGEST_A,
        factory_reference=_FACTORY_REF,
    )
    scope_prefix = _artifact_scope_prefix(digest)
    assert any(name.startswith(scope_prefix) for name in sys.modules)
    assert "example_agent.factory" not in sys.modules
    assert "example_agent.config" not in sys.modules
    assert sys.modules["example_agent"].__doc__ == (
        "Intergrax artifact-scoped top-level package dispatch stub."
    )


def test_concurrent_nn1_resolution_is_deterministic(tmp_path: Path) -> None:
    artifact_n, digest_n, lock_n = _build_artifact(tmp_path / "cn", marker="CN")
    artifact_n1, digest_n1, lock_n1 = _build_artifact(
        tmp_path / "cn1",
        marker="CN1",
        package_digest=_DIGEST_B,
    )
    revision_n = _revision(
        "rev-cn",
        artifact_digest=digest_n,
        lock_id=lock_n.lock_id or "",
        lock_digest=lock_n.lock_digest or "",
    )
    revision_n1 = _revision(
        "rev-cn1",
        artifact_digest=digest_n1,
        lock_id=lock_n1.lock_id or "",
        lock_digest=lock_n1.lock_digest or "",
        package_digests=(_DIGEST_B,),
    )
    resolver_n = _resolver(artifact_n, digest_n)
    resolver_n1 = _resolver(artifact_n1, digest_n1)

    def _resolve_n() -> str:
        factory = resolver_n.resolve_factory(
            runtime_revision=revision_n,
            package_digest=_DIGEST_A,
            factory_reference=_FACTORY_REF,
        )
        return _invoke_marker(factory, "n")

    def _resolve_n1() -> str:
        factory = resolver_n1.resolve_factory(
            runtime_revision=revision_n1,
            package_digest=_DIGEST_B,
            factory_reference=_FACTORY_REF,
        )
        return _invoke_marker(factory, "n1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        n_results = list(pool.map(lambda _: _resolve_n(), range(16)))
        n1_results = list(pool.map(lambda _: _resolve_n1(), range(16)))
    assert all(marker == "CN" for marker in n_results)
    assert all(marker == "CN1" for marker in n1_results)
