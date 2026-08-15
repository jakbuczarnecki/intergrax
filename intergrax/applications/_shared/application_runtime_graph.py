# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative Tier-3 runtime graph resolution (application → agents → platform).

Resolves the full acyclic transitive closure of Tier-2 agent dependencies.
Traversal: depth-first, first-discovery order for transitive agents; direct
agents retain application declaration order. Cycle detection uses
unvisited / visiting / visited states.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from intergrax.runtime_graph_semantics import (
    GraphVisitState,
    agent_dir_from_distribution,
    agent_distribution_name,
    format_agent_dependency_cycle,
    is_application_distribution,
    normalize_distribution_name,
    parse_dependency_name,
    parse_platform_extras,
)

__all__ = (
    "agent_dir_from_distribution",
    "agent_distribution_name",
    "parse_dependency_name",
    "normalize_distribution_name",
)


WorkspaceKind = Literal["platform", "agent", "application", "other"]


@dataclass(frozen=True)
class WorkspacePackage:
    """One recognized local workspace distribution."""

    dist_name: str
    kind: WorkspaceKind
    member_path: str
    dir_name: str


@dataclass(frozen=True)
class ApplicationRuntimeGraph:
    """Complete runtime graph for one Tier-3 application project."""

    application: str
    application_dist: str
    platform_extras: tuple[str, ...]

    direct_agent_dirs: tuple[str, ...]
    direct_agent_distributions: tuple[str, ...]

    transitive_agent_dirs: tuple[str, ...]
    transitive_agent_distributions: tuple[str, ...]

    all_agent_dirs: tuple[str, ...]
    all_agent_distributions: tuple[str, ...]

    direct_third_party_distributions: tuple[str, ...]
    workspace_member_paths: tuple[str, ...]

    @property
    def application_project_path(self) -> str:
        return f"applications/{self.application}"

    @property
    def agent_dirs(self) -> tuple[str, ...]:
        """Deprecated alias for ``all_agent_dirs`` (backward compatible)."""
        return self.all_agent_dirs

    @property
    def agent_distributions(self) -> tuple[str, ...]:
        """Deprecated alias for ``all_agent_distributions``."""
        return self.all_agent_distributions

    @property
    def third_party_distributions(self) -> tuple[str, ...]:
        """Deprecated alias for ``direct_third_party_distributions``."""
        return self.direct_third_party_distributions


def build_workspace_registry(repo_root: Path) -> dict[str, WorkspacePackage]:
    """Map normalized distribution names → workspace members.

    Distinguishes Intergrax-ai, Tier-2 agents, Tier-3 applications, and other
    recognized workspace packages. Naming conventions alone are not enough —
    the corresponding workspace member and project metadata must exist.
    """
    root_pyproject = repo_root / "pyproject.toml"
    if not root_pyproject.is_file():
        raise FileNotFoundError(
            f"RUNTIME_GRAPH_UNRESOLVED: missing root pyproject: {root_pyproject}"
        )
    root_data = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    registry: dict[str, WorkspacePackage] = {}

    root_name = str(root_data.get("project", {}).get("name", "Intergrax-ai"))
    registry[normalize_distribution_name(root_name)] = WorkspacePackage(
        dist_name=root_name,
        kind="platform",
        member_path=".",
        dir_name=".",
    )
    # Common alternate spellings for the platform package.
    for alias in ("Intergrax-ai", "intergrax-ai", "intergrax_ai"):
        registry.setdefault(
            normalize_distribution_name(alias),
            WorkspacePackage(
                dist_name=root_name,
                kind="platform",
                member_path=".",
                dir_name=".",
            ),
        )

    members = (
        root_data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    )
    for member in members:
        member_path = str(member).replace("\\", "/").strip().strip("/")
        pyproject = repo_root / member_path / "pyproject.toml"
        if not pyproject.is_file():
            raise FileNotFoundError(
                f"RUNTIME_GRAPH_UNRESOLVED: missing workspace member metadata: {pyproject}"
            )
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        dist_name = str(data.get("project", {}).get("name", "")).strip()
        if not dist_name:
            raise ValueError(
                f"RUNTIME_GRAPH_UNRESOLVED: workspace member {member_path} "
                "missing [project].name"
            )
        if member_path.startswith("agents/"):
            kind: WorkspaceKind = "agent"
            dir_name = member_path.split("/", 1)[1]
        elif member_path.startswith("applications/"):
            kind = "application"
            dir_name = member_path.split("/", 1)[1]
        else:
            kind = "other"
            dir_name = Path(member_path).name
        entry = WorkspacePackage(
            dist_name=dist_name,
            kind=kind,
            member_path=member_path,
            dir_name=dir_name,
        )
        key = normalize_distribution_name(dist_name)
        if key in registry and registry[key].kind != kind:
            raise ValueError(
                f"RUNTIME_GRAPH_UNRESOLVED: conflicting workspace entry for {dist_name}"
            )
        registry[key] = entry
    return registry


def _load_project(pyproject: Path) -> dict:
    if not pyproject.is_file():
        raise FileNotFoundError(
            f"RUNTIME_GRAPH_UNRESOLVED: missing project metadata: {pyproject}"
        )
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _workspace_sources(data: dict) -> dict:
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    return sources if isinstance(sources, dict) else {}


def _is_workspace_source(sources: dict, name: str) -> bool:
    for key, value in sources.items():
        if normalize_distribution_name(str(key)) != normalize_distribution_name(name):
            continue
        if isinstance(value, dict) and value.get("workspace"):
            return True
    return False


def _classify_dep(
    *,
    name: str,
    sources: dict,
    registry: dict[str, WorkspacePackage],
    importer_kind: Literal["application", "agent"],
    importer_label: str,
    importer_pyproject: Path,
) -> tuple[Literal["platform", "agent", "third_party"], WorkspacePackage | None]:
    entry = registry.get(normalize_distribution_name(name))
    if entry is not None and entry.kind == "platform":
        return "platform", entry
    if entry is not None and entry.kind == "agent":
        return "agent", entry
    if entry is not None and entry.kind == "application":
        if importer_kind == "agent":
            raise ValueError(
                "AGENT_TIER_VIOLATION: "
                f"importing agent {importer_label} depends on forbidden "
                f"application distribution {entry.dist_name} "
                f"(agent pyproject: {importer_pyproject})"
            )
        raise ValueError(
            f"APPLICATION_TIER_VIOLATION: {importer_label} depends on {entry.dist_name}"
        )
    if entry is not None and entry.kind == "other":
        raise ValueError(
            f"RUNTIME_GRAPH_UNRESOLVED: unrecognized local workspace package "
            f"{entry.dist_name!r} required by {importer_label} "
            f"({importer_pyproject})"
        )
    if _is_workspace_source(sources, name):
        raise ValueError(
            f"RUNTIME_GRAPH_UNRESOLVED: workspace source {name!r} on "
            f"{importer_label} is neither Intergrax-ai, a recognized Tier-2 agent, "
            f"nor the root application itself ({importer_pyproject})"
        )
    # Heuristic naming alone is insufficient without a registry entry.
    if agent_dir_from_distribution(name) is not None:
        raise ValueError(
            f"RUNTIME_GRAPH_UNRESOLVED: agent-like distribution {name!r} "
            f"is not a recognized workspace member (required by {importer_label})"
        )
    if is_application_distribution(name):
        if importer_kind == "agent":
            raise ValueError(
                "AGENT_TIER_VIOLATION: "
                f"importing agent {importer_label} depends on forbidden "
                f"application distribution {name} "
                f"(agent pyproject: {importer_pyproject})"
            )
        raise ValueError(
            f"APPLICATION_TIER_VIOLATION: {importer_label} depends on {name}"
        )
    return "third_party", None


def _ensure_agent_source(repo_root: Path, entry: WorkspacePackage) -> None:
    agent_path = repo_root / entry.member_path
    if not agent_path.is_dir():
        raise FileNotFoundError(
            f"RUNTIME_GRAPH_UNRESOLVED: missing agent source for "
            f"{entry.dist_name}: {agent_path}"
        )
    if not (agent_path / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"RUNTIME_GRAPH_UNRESOLVED: missing agent metadata for "
            f"{entry.dist_name}: {agent_path / 'pyproject.toml'}"
        )


def load_application_runtime_graph(
    repo_root: Path,
    application: str,
) -> ApplicationRuntimeGraph:
    """Derive the full transitive runtime graph for one application."""
    app_dir = application.strip().strip("/\\")
    app_pyproject = repo_root / "applications" / app_dir / "pyproject.toml"
    data = _load_project(app_pyproject)
    deps = list(data.get("project", {}).get("dependencies", []))
    app_dist = str(data.get("project", {}).get("name", ""))
    sources = _workspace_sources(data)
    registry = build_workspace_registry(repo_root)

    # Application self-entry must exist when present in workspace.
    app_entry = registry.get(normalize_distribution_name(app_dist)) if app_dist else None
    if app_entry is None:
        # Allow fixture trees that list the application as a member under a
        # matching path even if name lookup failed (should not happen).
        for entry in registry.values():
            if entry.kind == "application" and entry.dir_name == app_dir:
                app_entry = entry
                break

    extras: list[str] = []
    direct_agent_entries: list[WorkspacePackage] = []
    direct_third_party: list[str] = []
    seen_direct_dists: set[str] = set()

    for dep in deps:
        name = parse_dependency_name(dep)
        kind, entry = _classify_dep(
            name=name,
            sources=sources,
            registry=registry,
            importer_kind="application",
            importer_label=app_dir,
            importer_pyproject=app_pyproject,
        )
        if kind == "platform":
            extras.extend(parse_platform_extras(dep))
            continue
        if kind == "agent":
            assert entry is not None
            key = normalize_distribution_name(entry.dist_name)
            if key in seen_direct_dists:
                continue
            seen_direct_dists.add(key)
            _ensure_agent_source(repo_root, entry)
            direct_agent_entries.append(entry)
            continue
        direct_third_party.append(name)

    direct_keys = {normalize_distribution_name(e.dist_name) for e in direct_agent_entries}
    visit_state: dict[str, GraphVisitState] = {
        key: GraphVisitState.UNVISITED for key in registry if registry[key].kind == "agent"
    }
    # Also track agents discovered during traversal that may not yet be keyed.
    for entry in direct_agent_entries:
        visit_state.setdefault(normalize_distribution_name(entry.dist_name), GraphVisitState.UNVISITED)

    transitive_entries: list[WorkspacePackage] = []
    stack: list[str] = []

    def dfs(entry: WorkspacePackage) -> None:
        key = normalize_distribution_name(entry.dist_name)
        state = visit_state.get(key, GraphVisitState.UNVISITED)
        if state is GraphVisitState.VISITING:
            raise ValueError(format_agent_dependency_cycle(stack, entry.dist_name))
        if state is GraphVisitState.VISITED:
            return

        visit_state[key] = GraphVisitState.VISITING
        stack.append(entry.dist_name)
        _ensure_agent_source(repo_root, entry)
        agent_pyproject = repo_root / entry.member_path / "pyproject.toml"
        agent_data = _load_project(agent_pyproject)
        agent_deps = list(agent_data.get("project", {}).get("dependencies", []))
        agent_sources = _workspace_sources(agent_data)

        for dep in agent_deps:
            dep_name = parse_dependency_name(dep)
            dep_kind, dep_entry = _classify_dep(
                name=dep_name,
                sources=agent_sources,
                registry=registry,
                importer_kind="agent",
                importer_label=entry.dir_name,
                importer_pyproject=agent_pyproject,
            )
            if dep_kind != "agent":
                continue
            assert dep_entry is not None
            dep_key = normalize_distribution_name(dep_entry.dist_name)
            dep_state = visit_state.get(dep_key, GraphVisitState.UNVISITED)
            if dep_state is GraphVisitState.VISITING:
                raise ValueError(format_agent_dependency_cycle(stack, dep_entry.dist_name))
            if dep_state is GraphVisitState.VISITED:
                continue
            # First-discovery DFS: record transitive before descending only when
            # not already a direct application dependency.
            if dep_key not in direct_keys and all(
                normalize_distribution_name(t.dist_name) != dep_key for t in transitive_entries
            ):
                transitive_entries.append(dep_entry)
            dfs(dep_entry)

        stack.pop()
        visit_state[key] = GraphVisitState.VISITED

    for direct in direct_agent_entries:
        dfs(direct)

    direct_agent_dirs = tuple(e.dir_name for e in direct_agent_entries)
    direct_agent_distributions = tuple(e.dist_name for e in direct_agent_entries)
    transitive_agent_dirs = tuple(e.dir_name for e in transitive_entries)
    transitive_agent_distributions = tuple(e.dist_name for e in transitive_entries)
    all_agent_dirs = direct_agent_dirs + transitive_agent_dirs
    all_agent_distributions = direct_agent_distributions + transitive_agent_distributions

    member_paths = (
        f"applications/{app_dir}",
        *[f"agents/{d}" for d in all_agent_dirs],
    )
    return ApplicationRuntimeGraph(
        application=app_dir,
        application_dist=app_dist,
        platform_extras=tuple(dict.fromkeys(extras)),
        direct_agent_dirs=direct_agent_dirs,
        direct_agent_distributions=direct_agent_distributions,
        transitive_agent_dirs=transitive_agent_dirs,
        transitive_agent_distributions=transitive_agent_distributions,
        all_agent_dirs=all_agent_dirs,
        all_agent_distributions=all_agent_distributions,
        direct_third_party_distributions=tuple(direct_third_party),
        workspace_member_paths=member_paths,
    )


def list_application_projects(repo_root: Path) -> list[str]:
    apps = []
    root = repo_root / "applications"
    if not root.is_dir():
        return apps
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "pyproject.toml").is_file():
            apps.append(path.name)
    return apps


def list_agent_projects(repo_root: Path) -> list[str]:
    agents = []
    root = repo_root / "agents"
    if not root.is_dir():
        return agents
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "pyproject.toml").is_file():
            agents.append(path.name)
    return agents
