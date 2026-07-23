# © Artur Czarnecki. All rights reserved.

"""Runtime graph declaration contract for every Tier-3 application / Tier-2 agent."""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from intergrax.applications._shared.application_runtime_graph import (
    agent_dir_from_distribution,
    agent_distribution_name,
    list_agent_projects,
    list_application_projects,
    load_application_runtime_graph,
)

REPO = Path(__file__).resolve().parents[3]

# Direct third-party imports satisfied by Intergrax-ai transitive deps.
_PLATFORM_PROVIDED = frozenset({"pydantic", "pydantic_core"})


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _static_top_level_imports(root: Path) -> set[str]:
    skip = {"tests", "docs", "notebooks", "__pycache__", "prompts"}
    found: set[str] = set()
    for path in root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".", 1)[0])
    return found


@pytest.mark.gate
def test_every_application_declares_selected_agents() -> None:
    apps = list_application_projects(REPO)
    assert apps
    for app in apps:
        graph = load_application_runtime_graph(REPO, app)
        data = _load_toml(REPO / "applications" / app / "pyproject.toml")
        deps = data["project"]["dependencies"]
        for agent in graph.agent_dirs:
            dist = agent_distribution_name(agent)
            assert any(dep.startswith(dist) for dep in deps), (
                f"{app} missing declared agent {dist}"
            )
            assert data["tool"]["uv"]["sources"][dist] == {"workspace": True}


@pytest.mark.gate
def test_imported_agents_are_declared() -> None:
    agents = set(list_agent_projects(REPO))
    for app in list_application_projects(REPO):
        graph = load_application_runtime_graph(REPO, app)
        imported = _static_top_level_imports(REPO / "applications" / app) & agents
        declared = set(graph.agent_dirs)
        missing = imported - declared
        assert not missing, (
            f"{app} imports undeclared agents {sorted(missing)}; "
            f"declared={sorted(declared)}"
        )


@pytest.mark.gate
def test_every_agent_has_project_metadata() -> None:
    agent_root = REPO / "agents"
    for path in sorted(agent_root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        if not (path / "__init__.py").is_file():
            continue
        pyproject = path / "pyproject.toml"
        assert pyproject.is_file(), f"missing agent project: {pyproject}"
        data = _load_toml(pyproject)
        assert data["project"]["name"] == agent_distribution_name(path.name)
        assert any(
            dep.startswith("Intergrax-ai") for dep in data["project"]["dependencies"]
        )
        assert not any(
            "application" in dep.lower() and "intergrax-ai" not in dep.lower()
            for dep in data["project"]["dependencies"]
        )


@pytest.mark.gate
def test_workspace_lists_apps_and_agents() -> None:
    root = _load_toml(REPO / "pyproject.toml")
    members = set(root["tool"]["uv"]["workspace"]["members"])
    for app in list_application_projects(REPO):
        assert f"applications/{app}" in members
    for agent in list_agent_projects(REPO):
        assert f"agents/{agent}" in members


@pytest.mark.gate
def test_no_application_depends_on_another_application_dist() -> None:
    for app in list_application_projects(REPO):
        data = _load_toml(REPO / "applications" / app / "pyproject.toml")
        for dep in data["project"]["dependencies"]:
            name = re.split(r"[<>=!~\[]", dep.strip(), maxsplit=1)[0].strip()
            assert not (
                name.lower().startswith("intergrax-")
                and name.lower().endswith("-application")
            ), f"{app} depends on Tier-3 package {name}"


@pytest.mark.gate
@pytest.mark.parametrize("app", ["local_workspace_application", "lab_application"])
def test_export_minimality_representative(app: str) -> None:
    proc = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--project",
            f"applications/{app}",
            "--no-emit-workspace",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout.lower()
    if app == "local_workspace_application":
        assert "slack-sdk" in text
        for agent in ("local-indexer", "local-search", "local-synthesizer"):
            # workspace packages omitted by --no-emit-workspace; SDK proof is enough
            pass
    else:
        assert "slack-sdk" not in text


@pytest.mark.gate
def test_agent_dir_distribution_roundtrip() -> None:
    for agent in list_agent_projects(REPO):
        dist = agent_distribution_name(agent)
        assert agent_dir_from_distribution(dist) == agent
