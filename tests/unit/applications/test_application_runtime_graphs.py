# © Artur Czarnecki. All rights reserved.

"""Runtime graph declaration contract for every Tier-3 application / Tier-2 agent."""

from __future__ import annotations

import ast
import re
import subprocess
import textwrap
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
    skip = {"tests", "docs", "notebooks", "__pycache__", "prompts", "runtime-context"}
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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _agent_pyproject(
    *,
    dist: str,
    deps: list[str],
    workspace_sources: list[str] | None = None,
) -> str:
    dep_lines = ",\n".join(f'  "{d}"' for d in deps)
    sources = workspace_sources or [
        s for s in deps if s.startswith("intergrax-") or s == "Intergrax-ai"
    ]
    # Strip extras / markers for source keys
    source_keys: list[str] = []
    for s in sources:
        key = re.split(r"[<>=!~\[]", s.strip(), maxsplit=1)[0].strip()
        if key not in source_keys:
            source_keys.append(key)
    source_block = "\n".join(f'{k} = {{ workspace = true }}' for k in source_keys)
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


def _app_pyproject(*, dist: str, deps: list[str]) -> str:
    return _agent_pyproject(dist=dist, deps=deps)


def _build_fixture_repo(
    root: Path,
    *,
    app: str,
    app_dist: str,
    app_deps: list[str],
    agents: dict[str, list[str]],
    extra_apps: dict[str, tuple[str, list[str]]] | None = None,
    extra_workspace_members: list[str] | None = None,
) -> None:
    """Create an isolated mini-monorepo under ``root`` for graph tests."""
    members = [f"applications/{app}"]
    for agent_dir in agents:
        members.append(f"agents/{agent_dir}")
    if extra_apps:
        for other_app in extra_apps:
            members.append(f"applications/{other_app}")
    if extra_workspace_members:
        members.extend(extra_workspace_members)

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

    _write(
        root / "applications" / app / "pyproject.toml",
        _app_pyproject(dist=app_dist, deps=app_deps),
    )
    (root / "applications" / app / "__init__.py").write_text("", encoding="utf-8")

    for agent_dir, deps in agents.items():
        dist = agent_distribution_name(agent_dir)
        _write(
            root / "agents" / agent_dir / "pyproject.toml",
            _agent_pyproject(dist=dist, deps=deps),
        )
        (root / "agents" / agent_dir / "__init__.py").write_text("", encoding="utf-8")

    if extra_apps:
        for other_app, (other_dist, other_deps) in extra_apps.items():
            _write(
                root / "applications" / other_app / "pyproject.toml",
                _app_pyproject(dist=other_dist, deps=other_deps),
            )
            (root / "applications" / other_app / "__init__.py").write_text(
                "", encoding="utf-8"
            )


# ---------------------------------------------------------------------------
# Fleet gates (production repository)
# ---------------------------------------------------------------------------


@pytest.mark.gate
def test_every_application_declares_selected_agents() -> None:
    apps = list_application_projects(REPO)
    assert apps
    for app in apps:
        graph = load_application_runtime_graph(REPO, app)
        data = _load_toml(REPO / "applications" / app / "pyproject.toml")
        deps = data["project"]["dependencies"]
        for agent in graph.direct_agent_dirs:
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
        declared = set(graph.all_agent_dirs)
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


@pytest.mark.gate
def test_production_fleet_has_no_agent_to_agent_edges() -> None:
    edges: list[str] = []
    for agent in list_agent_projects(REPO):
        data = _load_toml(REPO / "agents" / agent / "pyproject.toml")
        for dep in data["project"]["dependencies"]:
            name = re.split(r"[<>=!~\[]", dep.strip(), maxsplit=1)[0].strip()
            if agent_dir_from_distribution(name) is not None:
                edges.append(f"{agent} → {name}")
    assert edges == [], f"unexpected production agent→agent edges: {edges}"


@pytest.mark.gate
@pytest.mark.parametrize(
    ("app", "expected_direct"),
    [
        (
            "local_workspace_application",
            ("local_indexer", "local_search", "local_synthesizer"),
        ),
        (
            "lab_application",
            ("echo", "lab", "problem_radar", "research", "signoff_probe"),
        ),
    ],
)
def test_representative_graphs_unchanged(
    app: str, expected_direct: tuple[str, ...]
) -> None:
    graph = load_application_runtime_graph(REPO, app)
    assert graph.direct_agent_dirs == expected_direct
    assert graph.transitive_agent_dirs == ()
    assert graph.all_agent_dirs == expected_direct


# ---------------------------------------------------------------------------
# Isolated transitive fixtures
# ---------------------------------------------------------------------------


@pytest.mark.gate
def test_one_level_transitive_dependency(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-agent-b-agent"],
            "agent_b": ["Intergrax-ai"],
            "agent_c": ["Intergrax-ai"],
        },
    )
    graph = load_application_runtime_graph(tmp_path, "example_application")
    assert graph.direct_agent_dirs == ("agent_a",)
    assert graph.transitive_agent_dirs == ("agent_b",)
    assert graph.all_agent_dirs == ("agent_a", "agent_b")
    assert graph.workspace_member_paths == (
        "applications/example_application",
        "agents/agent_a",
        "agents/agent_b",
    )


@pytest.mark.gate
def test_multi_level_transitive_dependency(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-agent-b-agent"],
            "agent_b": ["Intergrax-ai", "intergrax-agent-c-agent"],
            "agent_c": ["Intergrax-ai"],
        },
    )
    graph = load_application_runtime_graph(tmp_path, "example_application")
    assert graph.direct_agent_dirs == ("agent_a",)
    assert graph.transitive_agent_dirs == ("agent_b", "agent_c")
    assert graph.all_agent_dirs == ("agent_a", "agent_b", "agent_c")
    assert graph.workspace_member_paths == (
        "applications/example_application",
        "agents/agent_a",
        "agents/agent_b",
        "agents/agent_c",
    )


@pytest.mark.gate
def test_shared_transitive_deduplication(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=[
            "Intergrax-ai",
            "intergrax-agent-a-agent",
            "intergrax-agent-b-agent",
        ],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-agent-c-agent"],
            "agent_b": ["Intergrax-ai", "intergrax-agent-c-agent"],
            "agent_c": ["Intergrax-ai"],
        },
    )
    graph = load_application_runtime_graph(tmp_path, "example_application")
    assert graph.direct_agent_dirs == ("agent_a", "agent_b")
    assert graph.transitive_agent_dirs == ("agent_c",)
    assert graph.all_agent_dirs.count("agent_c") == 1


@pytest.mark.gate
def test_agent_dependency_cycle_fails_closed(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-agent-b-agent"],
            "agent_b": ["Intergrax-ai", "intergrax-agent-a-agent"],
        },
    )
    with pytest.raises(ValueError, match="AGENT_DEPENDENCY_CYCLE") as exc:
        load_application_runtime_graph(tmp_path, "example_application")
    msg = str(exc.value)
    assert "intergrax-agent-a-agent" in msg
    assert "intergrax-agent-b-agent" in msg
    assert "→" in msg


@pytest.mark.gate
def test_agent_to_application_fails_closed(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-example-application"],
        },
    )
    with pytest.raises(ValueError, match="AGENT_TIER_VIOLATION") as exc:
        load_application_runtime_graph(tmp_path, "example_application")
    msg = str(exc.value)
    assert "agent_a" in msg
    assert "intergrax-example-application" in msg


@pytest.mark.gate
def test_unknown_workspace_dependency_fails_closed(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=["Intergrax-ai", "intergrax-agent-a-agent"],
        agents={
            # workspace source is auto-emitted for intergrax-* deps; package is
            # not a workspace member → RUNTIME_GRAPH_UNRESOLVED
            "agent_a": ["Intergrax-ai", "intergrax-ghost-package"],
        },
    )
    with pytest.raises(ValueError, match="RUNTIME_GRAPH_UNRESOLVED"):
        load_application_runtime_graph(tmp_path, "example_application")


@pytest.mark.gate
def test_direct_also_reachable_transitively_stays_direct(tmp_path: Path) -> None:
    _build_fixture_repo(
        tmp_path,
        app="example_application",
        app_dist="intergrax-example-application",
        app_deps=[
            "Intergrax-ai",
            "intergrax-agent-a-agent",
            "intergrax-agent-b-agent",
        ],
        agents={
            "agent_a": ["Intergrax-ai", "intergrax-agent-b-agent"],
            "agent_b": ["Intergrax-ai"],
        },
    )
    graph = load_application_runtime_graph(tmp_path, "example_application")
    assert graph.direct_agent_dirs == ("agent_a", "agent_b")
    assert graph.transitive_agent_dirs == ()
    assert graph.all_agent_dirs == ("agent_a", "agent_b")
    assert graph.all_agent_dirs.count("agent_b") == 1


# ---------------------------------------------------------------------------
# Application discovery contract (artifact-based, name-agnostic)
# ---------------------------------------------------------------------------


def test_list_application_projects_discovers_manifest_contract(tmp_path: Path) -> None:
    _write(tmp_path / "applications" / "x7" / "manifest.py", "# contract marker\n")
    assert list_application_projects(tmp_path) == ["x7"]


def test_list_application_projects_ignores_non_contract_directories(
    tmp_path: Path,
) -> None:
    for name in ("build", "__pycache__", "helper"):
        (tmp_path / "applications" / name).mkdir(parents=True)
    assert list_application_projects(tmp_path) == []


def test_list_application_projects_is_name_independent(tmp_path: Path) -> None:
    _write(tmp_path / "applications" / "anything" / "manifest.py", "# contract marker\n")
    assert list_application_projects(tmp_path) == ["anything"]
    (tmp_path / "applications" / "package_only").mkdir(parents=True)
    _write(
        tmp_path / "applications" / "package_only" / "pyproject.toml",
        '[project]\nname = "pkg"\nversion = "0.1.0"\n',
    )
    assert list_application_projects(tmp_path) == ["anything"]
