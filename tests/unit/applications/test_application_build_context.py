# © Artur Czarnecki. All rights reserved.

"""Build-context materializer isolation gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from intergrax.applications._shared.application_build_context import (
    materialize_application_build_context,
)
from intergrax.applications._shared.application_runtime_graph import (
    list_application_projects,
    load_application_runtime_graph,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.gate
@pytest.mark.parametrize("app", ["local_workspace_application", "lab_application"])
def test_materialized_context_contains_only_selected_graph(
    app: str, tmp_path: Path
) -> None:
    out = tmp_path / "ctx"
    manifest = materialize_application_build_context(
        repo_root=REPO,
        application=app,
        output=out,
        pkg_port=8020 if "local" in app else 8090,
    )
    graph = load_application_runtime_graph(REPO, app)

    assert (out / "pyproject.toml").is_file()
    assert (out / "uv.lock").is_file()
    assert (out / "intergrax").is_dir()
    assert (out / "applications" / app).is_dir()
    assert (out / ".intergrax-runtime-graph.json").is_file()
    assert not (out / ".git").exists()
    assert not (out / ".venv").exists()
    assert not (out / "applications" / app / ".env").exists()

    present_apps = [
        p.name
        for p in (out / "applications").iterdir()
        if p.is_dir()
    ]
    assert present_apps == [app]

    present_agents = sorted(
        p.name for p in (out / "agents").iterdir() if p.is_dir()
    )
    assert present_agents == sorted(graph.agent_dirs)

    all_apps = set(list_application_projects(REPO))
    for other in sorted(all_apps - {app}):
        assert other in manifest["excluded_tier3_applications"]
        assert not (out / "applications" / other).exists()

    # Undeclared agents absent
    all_agents = {
        p.name
        for p in (REPO / "agents").iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }
    for undeclared in sorted(all_agents - set(graph.agent_dirs)):
        assert not (out / "agents" / undeclared).exists()

    stored = json.loads((out / ".intergrax-runtime-graph.json").read_text(encoding="utf-8"))
    assert stored["application"] == app
    assert stored["schema_version"] == 1
    assert ":" not in stored.get("lock_digest", "")  # no absolute paths


@pytest.mark.gate
def test_manifest_only_cli() -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build/build_application_image.py",
            "--application",
            "lab_application",
            "--manifest-only",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["application"] == "lab_application"
    assert "intergrax-echo-agent" in data["agent_packages"]
    assert "lab_application" not in data["excluded_tier3_applications"]
    assert "local_workspace_application" in data["excluded_tier3_applications"]


@pytest.mark.gate
def test_no_dockerfile_copies_all_applications_from_repo_root() -> None:
    """Committed compose must not use repository-root context."""
    for compose in (REPO / "applications").glob("*/docker/docker-compose*.yml"):
        text = compose.read_text(encoding="utf-8")
        assert "context: ../../.." not in text, compose
        if "build:" in text:
            assert "runtime-context" in text or "image:" in text, compose
