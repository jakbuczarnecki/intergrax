# © Artur Czarnecki. All rights reserved.

"""AP-8 runtime materialization adapter and filesystem safety tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agent_distribution.errors import MaterializationError
from intergrax.agent_distribution.materialization_adapters import (
    OciImageMaterializationAdapter,
    FakeRuntimeMaterializationAdapter,
)
from intergrax.agent_distribution.runtime_context_staging import (
    RUNTIME_GRAPH_MANIFEST_FILENAME,
    RUNTIME_LOCK_MANIFEST_FILENAME,
    resolve_safe_path,
    stage_graph_authorized_context,
    validate_materialization_output_root,
)
from tests.unit.agent_distribution.test_agent_distribution_materialization import (
    _build_fixture,
)

_DIGEST = "sha256:" + ("d" * 64)


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
    materialization_input, _adapter = _build_fixture(tmp_path)
    build_context = materialization_input.application_build_context
    candidate_dir = (
        Path(build_context.output_root)
        / f"candidate-{materialization_input.runtime_revision.runtime_revision_id}"
    )
    with pytest.raises(MaterializationError):
        stage_graph_authorized_context(
            build_context=build_context.model_copy(
                update={"agent_source_roots": (("missing-agent", "agents/missing"),)}
            ),
            graph=materialization_input.candidate_runtime_graph,
            lock=materialization_input.materialized_runtime_lock,
            effective_roster=materialization_input.effective_roster,
            application_release_id=build_context.application_release_id,
            candidate_dir=candidate_dir,
        )
    if candidate_dir.exists():
        assert not (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).exists()
    else:
        assert True


def test_test_adapter_stages_graph_authorized_closure(tmp_path: Path) -> None:
    materialization_input, _adapter = _build_fixture(tmp_path)
    adapter = FakeRuntimeMaterializationAdapter()
    output = adapter.materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    assert (candidate_dir / "applications" / "local_workspace_application" / "pyproject.toml").is_file()
    assert (candidate_dir / "agents" / "local_search_agent" / "pyproject.toml").is_file()
    assert (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).is_file()
    assert (candidate_dir / RUNTIME_LOCK_MANIFEST_FILENAME).is_file()
    assert output.materialization_artifact_digest.startswith("sha256:")
    assert output.runtime_graph_manifest_path == RUNTIME_GRAPH_MANIFEST_FILENAME


def test_lock_reference_embedded_in_staged_artifact(tmp_path: Path) -> None:
    materialization_input, _adapter = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter().materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    lock_payload = json.loads(
        (candidate_dir / RUNTIME_LOCK_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert lock_payload["lock_id"] == materialization_input.materialized_runtime_lock.lock_id


def test_oci_adapter_uses_injected_docker_runner(tmp_path: Path) -> None:
    materialization_input, _adapter = _build_fixture(tmp_path)
    calls: list[tuple[tuple[str, ...], Path]] = []

    def _fake_runner(args: tuple[str, ...], cwd: Path) -> str:
        calls.append((args, cwd))
        return f"intergrax/demo@sha256:{'e' * 64}"

    adapter = OciImageMaterializationAdapter(docker_build_runner=_fake_runner)
    output = adapter.materialize(materialization_input)
    assert calls
    assert calls[0][0][0] == "docker"
    assert "shell" not in " ".join(calls[0][0])
    assert output.materialization_artifact_digest == f"sha256:{'e' * 64}"
    assert output.topology.value == "oci_image"


def test_oci_build_failure_fail_closed(tmp_path: Path) -> None:
    materialization_input, _adapter = _build_fixture(tmp_path)

    def _fail_runner(args: tuple[str, ...], cwd: Path) -> str:
        raise MaterializationError("docker build failed: simulated")

    adapter = OciImageMaterializationAdapter(docker_build_runner=_fail_runner)
    with pytest.raises(MaterializationError):
        adapter.materialize(materialization_input)


def test_manifest_contains_enabled_agent_digests_without_secrets(tmp_path: Path) -> None:
    materialization_input, _adapter = _build_fixture(tmp_path)
    output = FakeRuntimeMaterializationAdapter().materialize(materialization_input)
    candidate_dir = Path(output.artifact_locator.removeprefix("test://"))
    manifest = json.loads(
        (candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["enabled_roster_agents"]
    assert all("package_digest" in item for item in manifest["enabled_roster_agents"])
    assert "xoxb-" not in json.dumps(manifest)
