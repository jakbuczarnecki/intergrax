# © Artur Czarnecki. All rights reserved.

"""AP-8 runtime materialization adapter and filesystem safety tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.dependency import (
    LockPackageRole,
    MaterializedAgentClosureEntry,
    MaterializedLockPackage,
)
from intergrax.agent_distribution.errors import (
    MaterializationError,
    MaterializationLockArtifactIdentityBlocked,
    MaterializationLockArtifactLocationBlocked,
)
from intergrax.agent_distribution.materialization_adapters import (
    OciImageMaterializationAdapter,
    FakeRuntimeMaterializationAdapter,
    _default_docker_build_runner,
    _normalize_image_digest,
)
from intergrax.agent_distribution.runtime_context_staging import (
    ARTIFACTS_STAGING_DIR,
    RUNTIME_GRAPH_MANIFEST_FILENAME,
    RUNTIME_INSTALL_MANIFEST_FILENAME,
    RUNTIME_LOCK_MANIFEST_FILENAME,
    build_lock_driven_install_plan,
    digest_staging_key,
    resolve_safe_path,
    stage_graph_authorized_context,
    validate_materialization_output_root,
)
from tests.unit.agent_distribution.test_agent_distribution_materialization import (
    _REQUESTS_DIGEST,
    _artifact_provider_for_wheel,
    _build_fixture,
    _AGENT_A,
)

_DIGEST = "sha256:" + ("d" * 64)
_DIGEST_E = "sha256:" + ("e" * 64)


def test_repository_root_output_refused(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    with pytest.raises(MaterializationError):
        validate_materialization_output_root(
            output_root=source_root,
            source_context_root=source_root,
        )


def test_traversal_path_refused(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    with pytest.raises(MaterializationError):
        resolve_safe_path(source_root, "../escape")


def test_partial_failure_does_not_publish_final_artifact(tmp_path: Path) -> None:
    materialization_input, _adapter, _artifact_provider, _digest = _build_fixture(tmp_path)
    from intergrax.agent_distribution.in_memory_stores import InMemoryAgentArtifactMetadataStore
    from intergrax.agent_distribution.package_artifact_provider import (
        FilesystemArtifactStoreRefResolver,
        MetadataBackedPackageArtifactProvider,
    )

    empty_provider = MetadataBackedPackageArtifactProvider(
        metadata_store=InMemoryAgentArtifactMetadataStore(),
        ref_resolver=FilesystemArtifactStoreRefResolver(root=tmp_path),
    )
    build_context = materialization_input.application_build_context
    candidate_dir = (
        Path(build_context.output_root)
        / f"candidate-{materialization_input.runtime_revision.runtime_revision_id}"
    )
    with pytest.raises(MaterializationLockArtifactLocationBlocked):
        stage_graph_authorized_context(
            build_context=build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=build_context.application_release_id,
            candidate_dir=candidate_dir,
            package_artifact_provider=empty_provider,
        )
    if candidate_dir.exists():
        assert not (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).exists()
    else:
        assert True


def test_test_adapter_stages_graph_authorized_closure(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, agent_digest = _build_fixture(tmp_path)
    adapter = FakeRuntimeMaterializationAdapter(package_artifact_provider=artifact_provider)
    output = adapter.materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    assert (candidate_dir / "applications" / "local_workspace_application" / "pyproject.toml").is_file()
    artifact_dir = candidate_dir / ARTIFACTS_STAGING_DIR / digest_staging_key(agent_digest)
    assert artifact_dir.is_dir()
    assert any(path.suffix == ".whl" for path in artifact_dir.iterdir())
    assert not (candidate_dir / "agents" / "local_search_agent").exists()
    assert (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).is_file()
    assert (candidate_dir / RUNTIME_LOCK_MANIFEST_FILENAME).is_file()
    assert (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).is_file()
    assert output.materialization_artifact_digest.startswith("sha256:")
    assert output.runtime_graph_manifest_path == RUNTIME_GRAPH_MANIFEST_FILENAME


def test_lock_reference_embedded_in_staged_artifact(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    lock_payload = json.loads(
        (candidate_dir / RUNTIME_LOCK_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert lock_payload["lock_id"] == materialization_input.materialized_runtime_lock.lock_id


def test_oci_adapter_uses_injected_docker_runner(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    calls: list[tuple[tuple[str, ...], Path, str]] = []

    def _fake_runner(args: tuple[str, ...], cwd: Path, image_ref: str) -> str:
        calls.append((args, cwd, image_ref))
        return _DIGEST_E

    adapter = OciImageMaterializationAdapter(
        package_artifact_provider=artifact_provider,
        docker_build_runner=_fake_runner,
    )
    output = adapter.materialize(materialization_input)
    assert calls
    assert calls[0][0][0] == "docker"
    image_tag = calls[0][2]
    assert image_tag == calls[0][0][-2]  # -t value, not build context "."
    assert "shell" not in " ".join(calls[0][0])
    assert output.materialization_artifact_digest == _DIGEST_E
    assert output.artifact_locator == image_tag
    assert output.topology.value == "oci_image"


def test_oci_build_failure_fail_closed(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)

    def _fail_runner(args: tuple[str, ...], cwd: Path, image_ref: str) -> str:
        raise MaterializationError("docker build failed: simulated")

    adapter = OciImageMaterializationAdapter(
        package_artifact_provider=artifact_provider,
        docker_build_runner=_fail_runner,
    )
    with pytest.raises(MaterializationError):
        adapter.materialize(materialization_input)


def test_manifest_contains_enabled_agent_digests_without_secrets(tmp_path: Path) -> None:
    materialization_input, _adapter, artifact_provider, _digest = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    manifest = json.loads(
        (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["enabled_roster_agents"]
    assert all("package_digest" in item for item in manifest["enabled_roster_agents"])
    assert "xoxb-" not in json.dumps(manifest)


def _staged_artifacts_from_candidate(lock: object, candidate_dir: Path) -> dict[str, str]:
    staged: dict[str, str] = {}
    seen: set[str] = set()
    for entry in lock.agent_closure:
        if entry.distribution_package_id in seen:
            continue
        seen.add(entry.distribution_package_id)
        key = digest_staging_key(entry.package_digest)
        artifact_dir = candidate_dir / ARTIFACTS_STAGING_DIR / key
        wheel = next(artifact_dir.glob("*.whl"))
        rel = f"{ARTIFACTS_STAGING_DIR}/{key}/{wheel.name}"
        staged[entry.distribution_package_id.lower().replace("_", "-")] = rel
    return staged


def _stage_fixture_context(
    tmp_path: Path,
) -> tuple[object, object, object, Path, dict[str, str], object]:
    materialization_input, _, artifact_provider, _digest = _build_fixture(tmp_path)
    build_context = materialization_input.application_build_context
    candidate_dir = (
        Path(build_context.output_root)
        / f"candidate-{materialization_input.runtime_revision.runtime_revision_id}"
    )
    staged = stage_graph_authorized_context(
        build_context=build_context,
        graph=materialization_input.candidate_runtime_graph,
        lock=materialization_input.materialized_runtime_lock,
        effective_roster=materialization_input.effective_roster,
        application_release_id=build_context.application_release_id,
        candidate_dir=candidate_dir,
        package_artifact_provider=artifact_provider,
    )
    staged_artifacts = _staged_artifacts_from_candidate(
        materialization_input.materialized_runtime_lock,
        candidate_dir,
    )
    return (
        materialization_input,
        build_context,
        staged,
        candidate_dir,
        staged_artifacts,
        artifact_provider,
    )


def test_oci_dockerfile_does_not_use_repository_uv_lock_authority(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, _digest = _build_fixture(tmp_path)

    def _fake_runner(args: tuple[str, ...], cwd: Path, image_ref: str) -> str:
        return _DIGEST_E

    OciImageMaterializationAdapter(
        package_artifact_provider=artifact_provider,
        docker_build_runner=_fake_runner,
    ).materialize(materialization_input)
    candidate_dir = (
        Path(materialization_input.application_build_context.output_root)
        / f"candidate-{materialization_input.runtime_revision.runtime_revision_id}"
    )
    dockerfile = (candidate_dir / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync" not in dockerfile
    assert "uv.lock" not in dockerfile
    assert RUNTIME_INSTALL_MANIFEST_FILENAME in dockerfile
    assert f"COPY {ARTIFACTS_STAGING_DIR}/" in dockerfile
    assert "COPY agents/" not in dockerfile


def test_install_manifest_derived_from_materialized_runtime_lock(tmp_path: Path) -> None:
    materialization_input, build_context, staged, candidate_dir, staged_artifacts, _ = (
        _stage_fixture_context(tmp_path)
    )
    install_text = (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    assert install_text == plan.manifest_text
    assert _AGENT_A in install_text
    assert "requests==2.32.0" in install_text
    assert f"--hash={_REQUESTS_DIGEST}" in install_text


def test_lock_package_absent_from_repo_deps_in_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, _, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    extra_digest = "sha256:" + ("f" * 64)
    extra = MaterializedLockPackage(
        distribution_name="orphan-extra-pkg",
        version="9.9.9",
        package_digest=extra_digest,
    )
    lock = materialization_input.materialized_runtime_lock.model_copy(
        update={
            "packages": (*materialization_input.materialized_runtime_lock.packages, extra),
            "lock_id": None,
            "lock_digest": None,
        }
    ).with_content_identity()
    plan = build_lock_driven_install_plan(
        lock=lock,
        build_context=build_context,
        staged_source_roots=("intergrax/",),
        staged_package_artifacts=staged_artifacts,
    )
    assert any(entry.distribution_name == "orphan-extra-pkg" for entry in plan.entries)
    assert f"orphan-extra-pkg==9.9.9 --hash={extra_digest}" in plan.manifest_text


def test_package_absent_from_lock_not_in_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    names = {entry.distribution_name for entry in plan.entries}
    assert "not-in-lock-package" not in names
    assert names == {pkg.distribution_name for pkg in materialization_input.materialized_runtime_lock.packages}


def test_exact_lock_version_preserved_in_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    requests = next(e for e in plan.entries if e.distribution_name == "requests")
    assert requests.version == "2.32.0"
    assert "requests==2.32.0" in requests.install_line


def test_lock_digest_preserved_when_supported(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    digest = "sha256:" + ("c" * 64)
    packages = tuple(
        MaterializedLockPackage(
            distribution_name=pkg.distribution_name,
            version=pkg.version,
            package_digest=digest if pkg.distribution_name == "requests" else pkg.package_digest,
        )
        for pkg in materialization_input.materialized_runtime_lock.packages
    )
    lock = materialization_input.materialized_runtime_lock.model_copy(
        update={
            "packages": packages,
            "lock_id": None,
            "lock_digest": None,
        }
    ).with_content_identity()
    plan = build_lock_driven_install_plan(
        lock=lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    requests = next(e for e in plan.entries if e.distribution_name == "requests")
    assert f"--hash={digest}" in requests.install_line


def test_graph_direct_agent_in_physical_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    agent = next(e for e in plan.entries if e.distribution_name == _AGENT_A)
    assert agent.source_kind == "artifact"
    assert ARTIFACTS_STAGING_DIR in agent.install_line
    assert "agents/local_search_agent" not in agent.install_line


def test_transitive_lock_package_in_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    assert any(entry.distribution_name == "requests" for entry in plan.entries)


def test_repository_uv_lock_change_does_not_alter_install_plan(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    baseline = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    source_root = Path(build_context.source_context_root)
    (source_root / "uv.lock").write_text("# mutated lock content\n", encoding="utf-8")
    repeated = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    assert repeated.manifest_text == baseline.manifest_text


def test_missing_artifact_location_fail_closed(tmp_path: Path) -> None:
    materialization_input, build_context, _, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    orphan_agent = MaterializedLockPackage(
        distribution_name="remote-only-agent",
        version="1.0.0",
        package_digest=_DIGEST,
    )
    lock = materialization_input.materialized_runtime_lock.model_copy(
        update={
            "packages": (
                *materialization_input.materialized_runtime_lock.packages,
                orphan_agent,
            ),
            "agent_closure": (
                *materialization_input.materialized_runtime_lock.agent_closure,
                MaterializedAgentClosureEntry(
                    distribution_package_id="remote-only-agent",
                    package_digest=_DIGEST,
                    role=LockPackageRole.DIRECT,
                ),
            ),
            "lock_id": None,
            "lock_digest": None,
        }
    ).with_content_identity()
    with pytest.raises(MaterializationLockArtifactLocationBlocked) as exc:
        build_lock_driven_install_plan(
            lock=lock,
            build_context=build_context,
            staged_source_roots=("intergrax/",),
            staged_package_artifacts=staged_artifacts,
        )
    assert MaterializationLockArtifactLocationBlocked.BLOCKER_CODE in str(exc.value)


def test_default_runner_inspects_image_tag_not_build_context(tmp_path: Path) -> None:
    image_ref = "intergrax/demo:candidate-test"

    def _fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, timeout=None):
        if cmd[0] == "docker" and cmd[1] == "build":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[1] == "inspect" and "{{index .RepoDigests 0}}" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, f"{image_ref}@sha256:{'f' * 64}", "")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=_fake_run):
        digest = _default_docker_build_runner(
            ["docker", "build", "-f", "Dockerfile", "-t", image_ref, "."],
            tmp_path,
            image_ref,
        )
    assert digest == f"sha256:{'f' * 64}"


def test_default_runner_repo_digest_normalized(tmp_path: Path) -> None:
    image_ref = "intergrax/demo:tag"
    repo_digest = f"registry.example/intergrax/demo@sha256:{'A' * 64}"

    def _fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, timeout=None):
        if cmd[1] == "build":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "{{index .RepoDigests 0}}" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, repo_digest, "")
        raise AssertionError(cmd)

    with patch("subprocess.run", side_effect=_fake_run):
        digest = _default_docker_build_runner(["docker", "build"], tmp_path, image_ref)
    assert digest == f"sha256:{'a' * 64}"


def test_default_runner_image_id_fallback_normalized(tmp_path: Path) -> None:
    image_ref = "intergrax/demo:tag"

    def _fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, timeout=None):
        if cmd[1] == "build":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "{{index .RepoDigests 0}}" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if "{{.Id}}" in cmd[2]:
            return subprocess.CompletedProcess(cmd, 0, f"sha256:{'b' * 64}", "")
        raise AssertionError(cmd)

    with patch("subprocess.run", side_effect=_fake_run):
        digest = _default_docker_build_runner(["docker", "build"], tmp_path, image_ref)
    assert digest == f"sha256:{'b' * 64}"


def test_default_runner_inspect_failure_fail_closed(tmp_path: Path) -> None:
    image_ref = "intergrax/demo:tag"

    def _fake_run(cmd, cwd=None, check=False, capture_output=False, text=False, timeout=None):
        if cmd[1] == "build":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "")

    import subprocess

    with patch("subprocess.run", side_effect=_fake_run):
        with pytest.raises(MaterializationError):
            _default_docker_build_runner(["docker", "build"], tmp_path, image_ref)


def test_normalize_image_digest_rejects_build_context_path() -> None:
    with pytest.raises(MaterializationError):
        _normalize_image_digest(".")


def test_artifact_digest_not_caller_supplied_on_oci_path(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, _digest = _build_fixture(tmp_path)

    def _fake_runner(args: tuple[str, ...], cwd: Path, image_ref: str) -> str:
        return _DIGEST_E

    output = OciImageMaterializationAdapter(
        package_artifact_provider=artifact_provider,
        docker_build_runner=_fake_runner,
    ).materialize(materialization_input)
    assert output.materialization_artifact_digest == _DIGEST_E
    assert output.artifact_locator != "."
    assert output.artifact_locator.startswith("intergrax/")


def test_lock_digest_with_matching_artifact_bytes_accepted(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, agent_digest = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    staged = candidate_dir / ARTIFACTS_STAGING_DIR / digest_staging_key(agent_digest)
    assert staged.is_dir()


def test_lock_digest_with_mismatched_artifact_bytes_fail_closed(tmp_path: Path) -> None:
    materialization_input, _, _, agent_digest = _build_fixture(tmp_path)
    wheel_path = tmp_path / "artifact-store" / "intergrax_local_search_agent-1.0.0-py3-none-any.whl"
    wheel_path.write_bytes(b"tampered-wheel-bytes")
    provider = _artifact_provider_for_wheel(tmp_path, wheel_path, agent_digest)
    with pytest.raises(MaterializationError, match="artifact digest mismatch"):
        stage_graph_authorized_context(
            build_context=materialization_input.application_build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=materialization_input.application_build_context.application_release_id,
            candidate_dir=Path(materialization_input.application_build_context.output_root) / "candidate-mismatch",
            package_artifact_provider=provider,
        )


def test_missing_agent_artifact_metadata_fail_closed(tmp_path: Path) -> None:
    materialization_input, _, _, _ = _build_fixture(tmp_path)
    from intergrax.agent_distribution.in_memory_stores import InMemoryAgentArtifactMetadataStore
    from intergrax.agent_distribution.package_artifact_provider import (
        FilesystemArtifactStoreRefResolver,
        MetadataBackedPackageArtifactProvider,
    )

    provider = MetadataBackedPackageArtifactProvider(
        metadata_store=InMemoryAgentArtifactMetadataStore(),
        ref_resolver=FilesystemArtifactStoreRefResolver(root=tmp_path),
    )
    with pytest.raises(MaterializationLockArtifactLocationBlocked):
        stage_graph_authorized_context(
            build_context=materialization_input.application_build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=materialization_input.application_build_context.application_release_id,
            candidate_dir=Path(materialization_input.application_build_context.output_root) / "candidate-missing",
            package_artifact_provider=provider,
        )


def test_artifact_provider_used_not_mutable_workspace_directory(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, agent_digest = _build_fixture(tmp_path)
    source_root = Path(materialization_input.application_build_context.source_context_root)
    mutable_agent_dir = source_root / "agents" / "local_search_agent"
    mutable_agent_dir.mkdir(parents=True, exist_ok=True)
    (mutable_agent_dir / "pyproject.toml").write_text("[project]\nname='tampered'\n", encoding="utf-8")
    output = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    assert not (candidate_dir / "agents").exists()
    install_text = (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert ARTIFACTS_STAGING_DIR in install_text
    assert digest_staging_key(agent_digest) in install_text


def test_staged_artifact_layout_is_digest_keyed_and_deterministic(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, agent_digest = _build_fixture(tmp_path)
    out_a = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    out_b = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    dir_a = Path(out_a.artifact_locator.removeprefix("test://")) / ARTIFACTS_STAGING_DIR
    dir_b = Path(out_b.artifact_locator.removeprefix("test://")) / ARTIFACTS_STAGING_DIR
    assert list(dir_a.rglob("*.whl"))[0].read_bytes() == list(dir_b.rglob("*.whl"))[0].read_bytes()
    assert (dir_a / digest_staging_key(agent_digest)).is_dir()


def test_third_party_without_package_digest_fail_closed(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    packages = tuple(
        MaterializedLockPackage(
            distribution_name=pkg.distribution_name,
            version=pkg.version,
            package_digest=None if pkg.distribution_name == "requests" else pkg.package_digest,
        )
        for pkg in materialization_input.materialized_runtime_lock.packages
    )
    lock = materialization_input.materialized_runtime_lock.model_copy(
        update={"packages": packages, "lock_id": None, "lock_digest": None}
    ).with_content_identity()
    with pytest.raises(MaterializationLockArtifactIdentityBlocked) as exc:
        build_lock_driven_install_plan(
            lock=lock,
            build_context=build_context,
            staged_source_roots=staged,
            staged_package_artifacts=staged_artifacts,
        )
    assert MaterializationLockArtifactIdentityBlocked.BLOCKER_CODE in str(exc.value)


def test_unrelated_workspace_source_does_not_alter_install_manifest(tmp_path: Path) -> None:
    materialization_input, build_context, staged, _, staged_artifacts, _ = _stage_fixture_context(
        tmp_path
    )
    baseline = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    source_root = Path(build_context.source_context_root)
    tampered = source_root / "agents" / "local_search_agent"
    tampered.mkdir(parents=True, exist_ok=True)
    (tampered / "pyproject.toml").write_text("[project]\nname='tampered'\n", encoding="utf-8")
    repeated = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    assert repeated.manifest_text == baseline.manifest_text


def test_uv_rejects_mismatched_index_package_hash(tmp_path: Path) -> None:
    req = tmp_path / "req.txt"
    req.write_text(
        "certifi==2024.2.2 --hash=sha256:"
        + ("0" * 64)
        + "\n",
        encoding="utf-8",
    )
    import subprocess

    venv = tmp_path / ".venv"
    subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True)
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "Scripts" / "python.exe"),
            "--no-deps",
            "-r",
            str(req),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Hash mismatch" in (result.stderr + result.stdout)


def test_builtin_agent_not_installed_from_mutable_source_directory(tmp_path: Path) -> None:
    materialization_input, build_context, staged, candidate_dir, staged_artifacts, _ = (
        _stage_fixture_context(tmp_path)
    )
    install_text = (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "agents/local_search_agent" not in install_text
    assert ARTIFACTS_STAGING_DIR in install_text
    plan = build_lock_driven_install_plan(
        lock=materialization_input.materialized_runtime_lock,
        build_context=build_context,
        staged_source_roots=staged,
        staged_package_artifacts=staged_artifacts,
    )
    agent = next(e for e in plan.entries if e.distribution_name == _AGENT_A)
    assert agent.source_kind == "artifact"


def test_builtin_agent_requires_immutable_artifact_identity(tmp_path: Path) -> None:
    materialization_input, _, _, _ = _build_fixture(tmp_path)
    from intergrax.agent_distribution.in_memory_stores import InMemoryAgentArtifactMetadataStore
    from intergrax.agent_distribution.package_artifact_provider import (
        FilesystemArtifactStoreRefResolver,
        MetadataBackedPackageArtifactProvider,
    )

    provider = MetadataBackedPackageArtifactProvider(
        metadata_store=InMemoryAgentArtifactMetadataStore(),
        ref_resolver=FilesystemArtifactStoreRefResolver(root=tmp_path),
    )
    with pytest.raises(MaterializationLockArtifactLocationBlocked):
        stage_graph_authorized_context(
            build_context=materialization_input.application_build_context,
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=materialization_input.application_build_context.application_release_id,
            candidate_dir=Path(materialization_input.application_build_context.output_root) / "candidate-builtin",
            package_artifact_provider=provider,
        )


def test_post_release_installed_agent_uses_exact_artifact_digest(tmp_path: Path) -> None:
    materialization_input, _, artifact_provider, agent_digest = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter(
        package_artifact_provider=artifact_provider
    ).materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    install_text = (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert digest_staging_key(agent_digest) in install_text
    manifest = json.loads(
        (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["enabled_roster_agents"][0]["package_digest"] == agent_digest
