# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative Tier-3 runtime graph resolution (application → agents → platform)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_AGENT_DIST_RE = re.compile(r"^intergrax-(.+)-agent$")
_EXTRA_RE = re.compile(r"Intergrax-ai(?:\[([^\]]*)\])?", re.IGNORECASE)


@dataclass(frozen=True)
class ApplicationRuntimeGraph:
    """Minimal runtime graph declared by one application project."""

    application: str
    application_dist: str
    platform_extras: tuple[str, ...]
    agent_dirs: tuple[str, ...]
    agent_distributions: tuple[str, ...]
    third_party_distributions: tuple[str, ...]
    workspace_member_paths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def application_project_path(self) -> str:
        return f"applications/{self.application}"


def agent_distribution_name(agent_dir: str) -> str:
    if agent_dir == "intergrax_assistant":
        return "intergrax-assistant-agent"
    return f"intergrax-{agent_dir.replace('_', '-')}-agent"


def agent_dir_from_distribution(dist: str) -> str | None:
    cleaned = dist.strip()
    if cleaned == "intergrax-assistant-agent":
        return "intergrax_assistant"
    match = _AGENT_DIST_RE.match(cleaned)
    if not match:
        return None
    return match.group(1).replace("-", "_")


def _dep_name(dep: str) -> str:
    raw = dep.strip().strip("\"'")
    raw = raw.split(";")[0].strip()
    raw = re.split(r"[<>=!~\[]", raw, maxsplit=1)[0].strip()
    return raw


def _parse_extras(dep: str) -> tuple[str, ...]:
    match = _EXTRA_RE.search(dep)
    if not match or not match.group(1):
        return ()
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def load_application_runtime_graph(
    repo_root: Path,
    application: str,
) -> ApplicationRuntimeGraph:
    """Derive the runtime graph solely from ``applications/<app>/pyproject.toml``."""
    app_dir = application.strip().strip("/\\")
    pyproject = repo_root / "applications" / app_dir / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"missing application project: {pyproject}")

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = list(data.get("project", {}).get("dependencies", []))
    app_dist = str(data.get("project", {}).get("name", ""))
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})

    extras: list[str] = []
    agent_dists: list[str] = []
    third_party: list[str] = []
    for dep in deps:
        name = _dep_name(dep)
        lower = name.lower()
        if lower in {"intergrax-ai", "intergrax_ai"}:
            extras.extend(_parse_extras(dep))
            continue
        agent_dir = agent_dir_from_distribution(name)
        if agent_dir is not None:
            agent_dists.append(name)
            continue
        if name in sources and isinstance(sources[name], dict) and sources[name].get(
            "workspace"
        ):
            # Workspace source that is not Intergrax-ai and not an agent dist → reject later.
            agent_dists.append(name)
            continue
        third_party.append(name)

    agent_dirs: list[str] = []
    for dist in agent_dists:
        folder = agent_dir_from_distribution(dist)
        if folder is None:
            raise ValueError(
                f"RUNTIME_GRAPH_UNRESOLVED: non-agent workspace dependency {dist!r} "
                f"on application {app_dir}"
            )
        agent_path = repo_root / "agents" / folder
        if not agent_path.is_dir():
            raise FileNotFoundError(
                f"RUNTIME_GRAPH_UNRESOLVED: missing agent source for {dist}: {agent_path}"
            )
        agent_dirs.append(folder)

    # Tier-3 → Tier-3 package dependencies are forbidden.
    for dep in deps:
        name = _dep_name(dep)
        if name.lower().startswith("intergrax-") and name.lower().endswith(
            "-application"
        ):
            raise ValueError(
                f"APPLICATION_TIER_VIOLATION: {app_dir} depends on {name}"
            )

    member_paths = (
        f"applications/{app_dir}",
        *[f"agents/{d}" for d in agent_dirs],
    )
    return ApplicationRuntimeGraph(
        application=app_dir,
        application_dist=app_dist,
        platform_extras=tuple(dict.fromkeys(extras)),
        agent_dirs=tuple(agent_dirs),
        agent_distributions=tuple(agent_dists),
        third_party_distributions=tuple(third_party),
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
