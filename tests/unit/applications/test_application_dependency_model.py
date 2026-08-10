# © Artur Czarnecki. All rights reserved.

"""Application dependency workspace contract (APPLICATION-DEPENDENCY-ARCHITECTURE-1)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
APPLICATIONS = (
    "attestation_demo",
    "dispute_sim_application",
    "governed_contractor_application",
    "intergrax_assistant_application",
    "lab_application",
    "legal_application",
    "local_workspace_application",
    "poc_template_application",
    "research_application",
)
LKW = "local_workspace_application"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.gate
def test_root_workspace_lists_all_real_applications() -> None:
    root = _load_toml(REPO / "pyproject.toml")
    members = root["tool"]["uv"]["workspace"]["members"]
    for name in APPLICATIONS:
        assert f"applications/{name}" in members
    deps = "\n".join(root["project"]["dependencies"])
    for name in APPLICATIONS:
        assert name not in deps
        assert f"intergrax-{name.replace('_', '-')}" not in deps


@pytest.mark.gate
@pytest.mark.parametrize("app_pkg", APPLICATIONS)
def test_application_pyproject_workspace_contract(app_pkg: str) -> None:
    path = REPO / "applications" / app_pkg / "pyproject.toml"
    assert path.is_file(), f"missing {path}"
    data = _load_toml(path)
    deps = data["project"]["dependencies"]
    assert any(dep.startswith("Intergrax-ai") for dep in deps)
    assert data["tool"]["uv"]["package"] is False
    assert data["tool"]["uv"]["sources"]["Intergrax-ai"] == {"workspace": True}
    assert any(
        dep.startswith("intergrax-") and dep.endswith("-agent") for dep in deps
    ), f"{app_pkg} must declare at least one Tier-2 agent package"

    dockerfile = (REPO / "applications" / app_pkg / "docker" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert f"--project applications/{app_pkg}" in dockerfile
    assert re.search(r"uv sync[^\n]*--extra ", dockerfile) is None
    assert "build_application_image.py" in dockerfile or "runtime-graph" in dockerfile.lower() or "materialized" in dockerfile.lower()
    assert "COPY agents/local_" not in dockerfile or "materialized" in dockerfile.lower()
    deploy = REPO / "applications" / app_pkg / "docs" / "BUILD_AND_DEPLOY.md"
    if not deploy.is_file() and app_pkg == "attestation_demo":
        deploy = REPO / "applications" / app_pkg / "BUILD_AND_DEPLOY.md"
    text = deploy.read_text(encoding="utf-8")
    assert "pyproject.toml" in text
    assert (
        "APPLICATION_DEPENDENCY_MODEL" in text
        or "APPLICATION_RUNTIME_GRAPH_MODEL" in text
        or "--project applications/" in text
        or "build_application_image.py" in text
    )
    assert "\x08" not in text
    assert "```bash" in text
    assert text.count("## Application dependency project") >= 1
    assert text.count("```") % 2 == 0

    ignore = (
        REPO / "applications" / app_pkg / "docker" / ".dockerignore"
    ).read_text(encoding="utf-8")
    assert ".git" in ignore
    assert ".venv" in ignore
    assert ".env" in ignore
    # No hand-written agent allowlists — graph comes from pyproject.
    assert "!agents/" not in ignore
    for line in ignore.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        assert not line.startswith(" ") and not line.startswith("\t"), (
            f"{app_pkg} .dockerignore has indented rule: {line!r}"
        )


@pytest.mark.gate
def test_scaffold_dependency_docs_have_valid_bash_fences(tmp_path: Path) -> None:
    from intergrax.scaffold.new_application import create_application

    target = create_application(
        name="dep_fence_scaffold",
        agents=["echo"],
        profile="lab",
        root=tmp_path,
        port=8192,
        force=True,
    )
    for rel in ("docs/BUILD_AND_DEPLOY.md", "docs/ARCHITECTURE.md"):
        text = (target / rel).read_text(encoding="utf-8")
        assert "\x08" not in text
        assert "```bash" in text
        assert text.count("## Application dependency project") == 1
        assert text.count("```") % 2 == 0
    pyproject = _load_toml(target / "pyproject.toml")
    assert any("echo-agent" in dep for dep in pyproject["project"]["dependencies"])
    dockerignore = (target / "docker" / ".dockerignore").read_text(encoding="utf-8")
    assert ".git" in dockerignore
    assert "!agents/" not in dockerignore


@pytest.mark.gate
def test_lkw_selects_slack_and_related_extras() -> None:
    data = _load_toml(REPO / "applications" / LKW / "pyproject.toml")
    joined = "\n".join(data["project"]["dependencies"])
    for extra in (
        "integrations-slack",
        "integrations-mongodb",
        "integrations-sentry",
        "integrations-kafka",
        "llm-ollama",
    ):
        assert extra in joined


@pytest.mark.gate
def test_lkw_runtime_graph_includes_ollama_client(tmp_path: Path) -> None:
    from intergrax.applications._shared.application_build_context import (
        materialize_application_build_context,
    )
    from intergrax.applications._shared.application_runtime_graph import (
        load_application_runtime_graph,
    )

    graph = load_application_runtime_graph(REPO, LKW)
    assert "llm-ollama" in graph.platform_extras

    out = tmp_path / "ctx"
    materialize_application_build_context(
        repo_root=REPO,
        application=LKW,
        output=out,
        pkg_port=8020,
    )
    lock_text = (out / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "ollama"' in lock_text


@pytest.mark.gate
def test_scaffold_emits_application_pyproject(tmp_path: Path) -> None:
    from intergrax.scaffold.new_application import create_application

    target = create_application(
        name="dep_model_scaffold",
        agents=["echo"],
        profile="lab",
        root=tmp_path,
        port=8191,
        force=True,
    )
    pyproject = target / "pyproject.toml"
    assert pyproject.is_file()
    data = _load_toml(pyproject)
    assert data["tool"]["uv"]["sources"]["Intergrax-ai"] == {"workspace": True}
    assert "intergrax-echo-agent" in data["tool"]["uv"]["sources"]
    dockerfile = (target / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert f"--project applications/{target.name}" in dockerfile
    assert "--extra " not in dockerfile.split("uv sync", 1)[-1].split("\n", 1)[0]
    compose = (target / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
    assert "runtime-context" in compose
    assert "context: ../../.." not in compose
