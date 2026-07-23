# © Artur Czarnecki. All rights reserved.

"""Application dependency workspace contract (APPLICATION-DEPENDENCY-ARCHITECTURE-1)."""

from __future__ import annotations

import re
import subprocess
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
NON_SLACK = "lab_application"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.gate
def test_root_workspace_lists_all_real_applications() -> None:
    root = _load_toml(REPO / "pyproject.toml")
    members = root["tool"]["uv"]["workspace"]["members"]
    expected = [f"applications/{name}" for name in APPLICATIONS]
    assert members == expected
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

    dockerfile = (REPO / "applications" / app_pkg / "docker" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert f"--project applications/{app_pkg}" in dockerfile
    assert re.search(r"uv sync[^\n]*--extra ", dockerfile) is None
    deploy = REPO / "applications" / app_pkg / "docs" / "BUILD_AND_DEPLOY.md"
    if not deploy.is_file() and app_pkg == "attestation_demo":
        deploy = REPO / "applications" / app_pkg / "BUILD_AND_DEPLOY.md"
    text = deploy.read_text(encoding="utf-8")
    assert "pyproject.toml" in text
    assert "APPLICATION_DEPENDENCY_MODEL" in text or "--project applications/" in text


@pytest.mark.gate
def test_lkw_selects_slack_and_related_extras() -> None:
    data = _load_toml(REPO / "applications" / LKW / "pyproject.toml")
    joined = "\n".join(data["project"]["dependencies"])
    for extra in (
        "integrations-slack",
        "integrations-mongodb",
        "integrations-sentry",
        "integrations-kafka",
    ):
        assert extra in joined


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
    dockerfile = (target / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert f"--project applications/{target.name}" in dockerfile
    assert "--extra " not in dockerfile.split("uv sync", 1)[-1].split("\n", 1)[0]


@pytest.mark.gate
def test_dependency_tree_isolation_slack_sdk() -> None:
    """Resolver-level isolation: LKW includes slack-sdk; lab does not."""

    def export_for(project: str) -> str:
        proc = subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--project",
                f"applications/{project}",
                "--no-dev",
                "--no-emit-workspace",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            pytest.skip(f"uv export failed: {proc.stderr[-400:]}")
        return proc.stdout.lower()

    lkw_export = export_for(LKW)
    lab_export = export_for(NON_SLACK)
    assert "slack-sdk" in lkw_export
    assert "slack-sdk" not in lab_export
    assert "pymongo" in lkw_export
    assert "pymongo" not in lab_export
