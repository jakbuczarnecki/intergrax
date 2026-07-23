# © Artur Czarnecki. All rights reserved.

"""Build-context materializer isolation gates."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from intergrax.applications._shared.application_build_context import (
    materialize_application_build_context,
)
from intergrax.applications._shared.application_runtime_graph import (
    agent_distribution_name,
    list_application_projects,
    load_application_runtime_graph,
)

REPO = Path(__file__).resolve().parents[3]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _agent_pyproject(*, dist: str, deps: list[str]) -> str:
    dep_lines = ",\n".join(f'  "{d}"' for d in deps)
    source_keys: list[str] = []
    for s in deps:
        key = s.split("[", 1)[0].strip()
        if key.startswith("intergrax") or key == "Intergrax-ai":
            if key not in source_keys:
                source_keys.append(key)
    source_block = "\n".join(f"{k} = {{ workspace = true }}" for k in source_keys)
    return f"""
    [project]
    name = "{dist}"
    version = "0.1.0"
    requires-python = ">=3.12,<3.13"
    dependencies = [
    {dep_lines},
    ]

    [build-system]
    requires = ["setuptools>=68"]
    build-backend = "setuptools.build_meta"

    [tool.uv.sources]
    {source_block}
    """


def _build_transitive_fixture(root: Path) -> None:
    app = "example_application"
    members = [
        f"applications/{app}",
        "applications/other_application",
        "agents/agent_a",
        "agents/agent_b",
        "agents/agent_c",
    ]
    member_lines = ",\n".join(f'  "{m}"' for m in members)
    _write(
        root / "pyproject.toml",
        f"""
        [project]
        name = "Intergrax-ai"
        version = "0.1.0"
        requires-python = ">=3.12,<3.13"
        dependencies = []

        [tool.uv.workspace]
        members = [
        {member_lines},
        ]
        """,
    )
    _write(root / "uv.lock", "# placeholder\n")
    _write(root / "README.md", "# fixture\n")
    _write(root / "intergrax" / "__init__.py", "# platform\n")

    _write(
        root / "applications" / app / "pyproject.toml",
        _agent_pyproject(
            dist="intergrax-example-application",
            deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        ),
    )
    _write(root / "applications" / app / "host.py", "APP = True\n")
    _write(
        root / "applications" / "other_application" / "pyproject.toml",
        _agent_pyproject(
            dist="intergrax-other-application",
            deps=["Intergrax-ai"],
        ),
    )
    _write(root / "applications" / "other_application" / "host.py", "OTHER = True\n")

    agents = {
        "agent_a": ["Intergrax-ai", "intergrax-agent-b-agent"],
        "agent_b": ["Intergrax-ai"],
        "agent_c": ["Intergrax-ai"],
    }
    for agent_dir, deps in agents.items():
        _write(
            root / "agents" / agent_dir / "pyproject.toml",
            _agent_pyproject(dist=agent_distribution_name(agent_dir), deps=deps),
        )
        _write(root / "agents" / agent_dir / "mod.py", f"NAME = {agent_dir!r}\n")


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
    assert present_agents == sorted(graph.all_agent_dirs)

    all_apps = set(list_application_projects(REPO))
    for other in sorted(all_apps - {app}):
        assert other in manifest["excluded_tier3_applications"]
        assert not (out / "applications" / other).exists()

    # Unreachable agents absent
    all_agents = {
        p.name
        for p in (REPO / "agents").iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }
    for undeclared in sorted(all_agents - set(graph.all_agent_dirs)):
        assert not (out / "agents" / undeclared).exists()

    stored = json.loads((out / ".intergrax-runtime-graph.json").read_text(encoding="utf-8"))
    assert stored["application"] == app
    assert stored["schema_version"] == 2
    assert stored["direct_agent_packages"] == list(graph.direct_agent_distributions)
    assert stored["transitive_agent_packages"] == list(
        graph.transitive_agent_distributions
    )
    assert stored["all_agent_packages"] == list(graph.all_agent_distributions)
    assert "direct_third_party_distributions" in stored
    assert "third_party_distributions" not in stored
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
    assert data["schema_version"] == 2
    assert "intergrax-echo-agent" in data["all_agent_packages"]
    assert "intergrax-echo-agent" in data["direct_agent_packages"]
    assert data["transitive_agent_packages"] == []
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


@pytest.mark.gate
def test_transitive_agents_materialized_and_unreachable_excluded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _build_transitive_fixture(repo)
    out = tmp_path / "ctx"
    manifest = materialize_application_build_context(
        repo_root=repo,
        application="example_application",
        output=out,
        pkg_port=8000,
    )

    assert (out / "applications" / "example_application").is_dir()
    assert (out / "agents" / "agent_a").is_dir()
    assert (out / "agents" / "agent_b").is_dir()
    assert not (out / "agents" / "agent_c").exists()
    assert not (out / "applications" / "other_application").exists()

    assert manifest["schema_version"] == 2
    assert manifest["direct_agent_packages"] == ["intergrax-agent-a-agent"]
    assert manifest["transitive_agent_packages"] == ["intergrax-agent-b-agent"]
    assert manifest["all_agent_packages"] == [
        "intergrax-agent-a-agent",
        "intergrax-agent-b-agent",
    ]
    assert manifest["direct_third_party_distributions"] == []
    assert "third_party_distributions" not in manifest


@pytest.mark.gate
def test_unexpected_agent_directory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from intergrax.applications._shared import application_build_context as mod

    repo = tmp_path / "repo"
    _build_transitive_fixture(repo)
    out = tmp_path / "ctx"
    real_copy = mod._copy_filtered_tree

    def sneaky_copy(src: Path, dst: Path, **kwargs):  # type: ignore[no-untyped-def]
        result = real_copy(src, dst, **kwargs)
        if src.name == "agent_a":
            stray = dst.parent / "agent_c"
            stray.mkdir(exist_ok=True)
            (stray / "x.py").write_text("X = 1\n", encoding="utf-8")
        return result

    monkeypatch.setattr(mod, "_copy_filtered_tree", sneaky_copy)
    with pytest.raises(ValueError, match="DOCKER_ISOLATION_FAILED"):
        materialize_application_build_context(
            repo_root=repo,
            application="example_application",
            output=out,
            pkg_port=8000,
        )
