# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph-authoritative runtime context staging helpers (AP-8 §19)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.errors import MaterializationError
from intergrax.agent_distribution.materialization import ApplicationBuildContext
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_graph import CandidateApplicationRuntimeGraph

RUNTIME_GRAPH_MANIFEST_FILENAME = ".intergrax-runtime-graph.json"
RUNTIME_LOCK_MANIFEST_FILENAME = ".intergrax-runtime-lock.json"

_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "node_modules",
        "dist",
        "build",
        ".build-context",
        "runtime-context",
        "proof_artifacts",
    }
)
_SKIP_FILE_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".DS_Store",
        "Thumbs.db",
    }
)
_SKIP_SUFFIXES = frozenset({".pyc", ".pyo", ".egg-info"})

_SECRET_PATTERNS = (
    re.compile(r"\bxapp-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bxoxb-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)Authorization:\s*Bearer\s+\S+"),
    re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_PLACEHOLDER_OK = re.compile(
    r"(?i)(changeme|placeholder|your[_-]|xxx|redacted|example|dummy|\$\{|<.+>)"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def resolve_safe_path(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; reject traversal escapes."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise MaterializationError(f"absolute path rejected: {relative}")
    if ".." in candidate.parts:
        raise MaterializationError(f"path traversal rejected: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterializationError(f"path escapes output root: {relative}") from exc
    return resolved


def validate_materialization_output_root(
    *,
    output_root: Path,
    source_context_root: Path,
) -> Path:
    """Ensure output root is explicit, bounded, and not the repository root."""
    output_root = output_root.resolve()
    source_context_root = source_context_root.resolve()
    if output_root == source_context_root:
        raise MaterializationError("refusing to materialize into repository root")
    if ".." in output_root.parts:
        raise MaterializationError("output root traversal rejected")
    return output_root


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_symlink_escape(repo_root: Path, path: Path) -> None:
    if path.is_symlink():
        target = path.resolve()
        if not _is_within(repo_root, target):
            raise MaterializationError(f"symlink escape rejected: {path} -> {target}")


def scan_secrets(path: Path, content: bytes) -> None:
    """Fail closed when non-placeholder secrets appear in staged build files."""
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".py"}:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return
        for pattern in _SECRET_PATTERNS[:3]:
            if pattern.search(text) and not _PLACEHOLDER_OK.search(text):
                raise MaterializationError(
                    f"secret-like content rejected in build context: {path}"
                )
        return
    sensitive_names = {
        ".env",
        ".env.local",
        ".env.production",
        ".pem",
        ".key",
        "credentials.json",
        "secrets.yaml",
        "secrets.yml",
        "secrets.toml",
    }
    name_l = path.name.lower()
    suffix = path.suffix.lower()
    if name_l not in sensitive_names and suffix not in {".env", ".pem", ".key", ".p12", ".pfx"}:
        if not name_l.endswith(".env"):
            return
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return
    for pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            snippet = match.group(0)
            if _PLACEHOLDER_OK.search(snippet):
                continue
            raise MaterializationError(
                f"secret-like content rejected in build context: {path}"
            )


def _copy_filtered_tree(
    src: Path,
    dst: Path,
    *,
    repo_root: Path,
    extra_skip_dirs: frozenset[str] | None = None,
) -> list[str]:
    included: list[str] = []
    skip_dirs = _SKIP_DIR_NAMES | (extra_skip_dirs or frozenset())
    if not src.exists():
        raise MaterializationError(f"missing authorized source root: {src}")
    _reject_symlink_escape(repo_root, src)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        current = Path(dirpath)
        _reject_symlink_escape(repo_root, current)
        dirnames[:] = [
            d for d in dirnames if d not in skip_dirs and not d.endswith(".egg-info")
        ]
        rel_dir = current.relative_to(src)
        for name in filenames:
            source_file = current / name
            if name in _SKIP_FILE_NAMES or source_file.suffix in _SKIP_SUFFIXES:
                continue
            if name.endswith(".egg-info"):
                continue
            _reject_symlink_escape(repo_root, source_file)
            target = dst / rel_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source_file.read_bytes()
            scan_secrets(source_file, data)
            target.write_bytes(data)
            included.append((rel_dir / name).as_posix())
    return included


def authorized_source_roots_for_graph(
    *,
    graph: CandidateApplicationRuntimeGraph,
    build_context: ApplicationBuildContext,
) -> tuple[str, ...]:
    """Return relative source roots authorized by graph closure + build context."""
    agent_roots = {package_id: rel_path for package_id, rel_path in build_context.agent_source_roots}
    required_agents = {
        agent.distribution_package_id for agent in (*graph.direct_agents, *graph.transitive_agents)
    }
    missing = sorted(required_agents - set(agent_roots))
    if missing:
        raise MaterializationError(
            "build context missing agent source roots for graph agents: "
            + ", ".join(missing)
        )
    roots = [
        build_context.application_source_root,
        *sorted(agent_roots[agent_id] for agent_id in required_agents),
    ]
    for root_name in ("pyproject.toml", "uv.lock"):
        if (Path(build_context.source_context_root) / root_name).is_file():
            roots.append(root_name)
    platform_root = Path(build_context.source_context_root) / "intergrax"
    if platform_root.is_dir():
        roots.append("intergrax/")
    return tuple(dict.fromkeys(roots))


def render_distribution_runtime_graph_manifest(
    *,
    graph: CandidateApplicationRuntimeGraph,
    lock: MaterializedRuntimeLock,
    application_release_id: str,
    effective_roster: EffectiveRoster,
    included_source_roots: tuple[str, ...],
) -> dict[str, Any]:
    """Serialize distribution runtime graph manifest for artifact embedding."""
    if graph.runtime_graph_digest is None:
        raise MaterializationError("graph must have content identity before manifest render")
    if lock.lock_id is None:
        raise MaterializationError("lock must have content identity before manifest render")

    enabled_agents = [
        {
            "logical_agent_id": entry.logical_agent_id,
            "distribution_package_id": entry.distribution_package_id,
            "package_digest": entry.package_digest,
        }
        for entry in effective_roster.entries
        if entry.effective_enablement
    ]
    graph_agents = [
        {
            "logical_agent_id": agent.logical_agent_id,
            "distribution_package_id": agent.distribution_package_id,
            "package_digest": agent.package_digest,
        }
        for agent in (*graph.direct_agents, *graph.transitive_agents)
    ]
    return {
        "schema_version": graph.graph_schema_version,
        "distribution_manifest_schema": "distribution_runtime_graph_manifest.v1",
        "application_id": graph.application_id,
        "application_release_id": application_release_id,
        "runtime_graph_digest": graph.runtime_graph_digest,
        "materialized_runtime_lock_id": lock.lock_id,
        "materialized_runtime_lock_digest": lock.lock_digest,
        "direct_agents": graph_agents[: len(graph.direct_agents)],
        "transitive_agents": graph_agents[len(graph.direct_agents) :],
        "enabled_roster_agents": enabled_agents,
        "direct_third_party_distributions": [
            item.model_dump(mode="json")
            for item in graph.direct_third_party_distributions
        ],
        "included_source_roots": list(included_source_roots),
    }


def render_runtime_lock_manifest(lock: MaterializedRuntimeLock) -> dict[str, Any]:
    """Embed canonical lock JSON for offline audit."""
    return lock.model_dump(mode="json")


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def directory_content_digest(root: Path) -> str:
    """Deterministic sha256 over relative file paths and bytes."""
    hasher = hashlib.sha256()
    if not root.is_dir():
        raise MaterializationError("artifact digest requires a directory root")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def stage_graph_authorized_context(
    *,
    build_context: ApplicationBuildContext,
    graph: CandidateApplicationRuntimeGraph,
    lock: MaterializedRuntimeLock,
    effective_roster: EffectiveRoster,
    application_release_id: str,
    candidate_dir: Path,
) -> tuple[str, ...]:
    """Stage minimal graph-authorized source closure into ``candidate_dir``."""
    source_root = Path(_strip_required(build_context.source_context_root)).resolve()
    candidate_dir = validate_materialization_output_root(
        output_root=candidate_dir,
        source_context_root=source_root,
    )
    included_roots = authorized_source_roots_for_graph(
        graph=graph,
        build_context=build_context,
    )
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    staged: list[str] = []

    for rel in included_roots:
        src = resolve_safe_path(source_root, rel.rstrip("/"))
        if src.is_file():
            data = src.read_bytes()
            scan_secrets(src, data)
            target = candidate_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            staged.append(rel)
            continue
        dst = candidate_dir / rel.rstrip("/")
        _copy_filtered_tree(
            src,
            dst,
            repo_root=source_root,
            extra_skip_dirs=frozenset({"tests", "docs", "notebooks", "proof", "docker"}),
        )
        staged.append(rel if rel.endswith("/") else f"{rel}/")

    manifest = render_distribution_runtime_graph_manifest(
        graph=graph,
        lock=lock,
        application_release_id=application_release_id,
        effective_roster=effective_roster,
        included_source_roots=tuple(sorted(set(staged))),
    )
    write_json_artifact(candidate_dir / RUNTIME_GRAPH_MANIFEST_FILENAME, manifest)
    write_json_artifact(
        candidate_dir / RUNTIME_LOCK_MANIFEST_FILENAME,
        render_runtime_lock_manifest(lock),
    )
    return tuple(sorted(set(staged)))
