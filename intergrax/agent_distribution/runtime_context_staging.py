# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Graph-authoritative runtime context staging helpers (AP-8 §19)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.errors import (
    MaterializationError,
    MaterializationLockArtifactIdentityBlocked,
    MaterializationLockArtifactLocationBlocked,
)
from intergrax.agent_distribution.materialization import ApplicationBuildContext
from intergrax.agent_distribution.package_artifact_provider import (
    PackageArtifactProvider,
    ResolvedPackageArtifact,
    verify_artifact_file_digest,
)
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_graph import CandidateApplicationRuntimeGraph

RUNTIME_GRAPH_MANIFEST_FILENAME = ".intergrax-runtime-graph.json"
RUNTIME_LOCK_MANIFEST_FILENAME = ".intergrax-runtime-lock.json"
RUNTIME_INSTALL_MANIFEST_FILENAME = ".intergrax-runtime-install.txt"
ARTIFACTS_STAGING_DIR = ".intergrax-artifacts"

_PYPROJECT_NAME_RE = re.compile(
    r"""(?:^|\n)\s*name\s*=\s*(['"])([^'"]+)\1""",
    re.MULTILINE,
)

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
    _ = graph
    roots = [
        build_context.application_source_root,
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


def _normalize_distribution_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _read_pyproject_distribution_name(pyproject_path: Path) -> str | None:
    if not pyproject_path.is_file():
        return None
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _PYPROJECT_NAME_RE.search(text)
    if match is None:
        return None
    return match.group(2).strip()


def _pip_hash_suffix(package_digest: str) -> str:
    digest = package_digest.strip().lower()
    if not digest.startswith("sha256:"):
        raise MaterializationError("package_digest must be sha256:<hex>")
    return f" --hash={digest}"


def digest_staging_key(package_digest: str) -> str:
    """Map lock digest to deterministic artifact staging directory name."""
    digest = package_digest.strip().lower()
    if not digest.startswith("sha256:"):
        raise MaterializationError("package_digest must be sha256:<hex>")
    return digest.replace(":", "-", 1)


def stage_verified_package_artifact(
    *,
    artifact: ResolvedPackageArtifact,
    candidate_dir: Path,
) -> str:
    """Copy one digest-verified package artifact into digest-keyed staging layout."""
    verify_artifact_file_digest(artifact.local_source_path, artifact.package_digest)
    staging_key = digest_staging_key(artifact.package_digest)
    dest_dir = candidate_dir / ARTIFACTS_STAGING_DIR / staging_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / artifact.local_source_path.name
    dest.write_bytes(artifact.local_source_path.read_bytes())
    verify_artifact_file_digest(dest, artifact.package_digest)
    return f"{ARTIFACTS_STAGING_DIR}/{staging_key}/{artifact.local_source_path.name}"


def stage_lock_authorized_package_artifacts(
    *,
    lock: MaterializedRuntimeLock,
    candidate_dir: Path,
    package_artifact_provider: PackageArtifactProvider,
) -> dict[str, str]:
    """Stage digest-verified agent/package artifacts authorized by lock closure."""
    staged: dict[str, str] = {}
    seen: set[str] = set()
    for entry in lock.agent_closure:
        if entry.distribution_package_id in seen:
            continue
        seen.add(entry.distribution_package_id)
        artifact = package_artifact_provider.resolve_artifact(
            entry.distribution_package_id,
            entry.package_digest,
        )
        rel_path = stage_verified_package_artifact(
            artifact=artifact,
            candidate_dir=candidate_dir,
        )
        staged[_normalize_distribution_name(entry.distribution_package_id)] = rel_path
    return staged


def validate_lock_package_install_completeness(
    *,
    lock: MaterializedRuntimeLock,
    build_context: ApplicationBuildContext,
    staged_source_roots: tuple[str, ...],
    staged_package_artifacts: dict[str, str],
) -> None:
    """Ensure every lock package has a deterministic production install source."""
    source_root = Path(build_context.source_context_root).resolve()
    app_root = build_context.application_source_root.rstrip("/")
    app_distribution = _read_pyproject_distribution_name(
        source_root / app_root / "pyproject.toml"
    )
    root_distribution = _read_pyproject_distribution_name(source_root / "pyproject.toml")
    staged_norm = {_normalize_distribution_name(root.rstrip("/")) for root in staged_source_roots}
    platform_staged = "intergrax" in staged_norm or any(
        norm.startswith("intergrax") for norm in staged_norm
    )
    closure_agent_ids = {
        _normalize_distribution_name(entry.distribution_package_id)
        for entry in lock.agent_closure
    }

    for package in lock.packages:
        norm = _normalize_distribution_name(package.distribution_name)
        has_artifact = norm in staged_package_artifacts
        has_app_path = (
            app_distribution is not None
            and norm == _normalize_distribution_name(app_distribution)
        )
        has_platform_path = (
            platform_staged
            and root_distribution is not None
            and norm == _normalize_distribution_name(root_distribution)
        )
        has_index_digest = package.package_digest is not None

        if has_artifact or has_app_path or has_platform_path:
            continue
        if norm in closure_agent_ids:
            raise MaterializationLockArtifactLocationBlocked(
                MaterializationLockArtifactLocationBlocked.BLOCKER_CODE
                + f": agent package {package.distribution_name} lacks verified artifact"
            )
        if not has_index_digest:
            raise MaterializationLockArtifactIdentityBlocked(
                MaterializationLockArtifactIdentityBlocked.BLOCKER_CODE
                + f": package {package.distribution_name} lacks package_digest"
            )


@dataclass(frozen=True)
class LockDrivenInstallEntry:
    """One lock-authoritative install line for OCI/runtime materialization."""

    distribution_name: str
    version: str
    install_line: str
    package_digest: str | None = None
    source_kind: str = "index"


@dataclass(frozen=True)
class LockDrivenInstallPlan:
    """Deterministic install manifest derived only from MaterializedRuntimeLock."""

    entries: tuple[LockDrivenInstallEntry, ...]
    manifest_text: str


def build_lock_driven_install_plan(
    *,
    lock: MaterializedRuntimeLock,
    build_context: ApplicationBuildContext,
    staged_source_roots: tuple[str, ...],
    staged_package_artifacts: dict[str, str] | None = None,
) -> LockDrivenInstallPlan:
    """Build exact install manifest from lock closure — repository uv.lock is not authority."""
    if lock.lock_id is None:
        raise MaterializationError("lock must have content identity before install plan render")

    source_root = Path(build_context.source_context_root).resolve()
    artifact_paths = staged_package_artifacts or {}
    closure_agent_ids = {
        _normalize_distribution_name(entry.distribution_package_id)
        for entry in lock.agent_closure
    }
    app_root = build_context.application_source_root.rstrip("/")
    app_distribution = _read_pyproject_distribution_name(
        source_root / app_root / "pyproject.toml"
    )
    root_distribution = _read_pyproject_distribution_name(source_root / "pyproject.toml")
    staged_norm = {_normalize_distribution_name(root.rstrip("/")) for root in staged_source_roots}
    platform_staged = "intergrax" in staged_norm or any(
        norm.startswith("intergrax") for norm in staged_norm
    )

    entries: list[LockDrivenInstallEntry] = []
    for package in sorted(
        lock.packages,
        key=lambda item: (
            _normalize_distribution_name(item.distribution_name),
            item.version,
            item.package_digest or "",
        ),
    ):
        norm = _normalize_distribution_name(package.distribution_name)
        install_line: str | None = None
        source_kind = "index"

        if norm in artifact_paths:
            install_line = (
                f"{package.distribution_name} @ file:///app/{artifact_paths[norm]}"
            )
            source_kind = "artifact"
        elif app_distribution is not None and norm == _normalize_distribution_name(
            app_distribution
        ):
            install_line = f"{package.distribution_name} @ file:///app/{app_root}"
            source_kind = "path"
        elif (
            platform_staged
            and root_distribution is not None
            and norm == _normalize_distribution_name(root_distribution)
        ):
            install_line = f"{package.distribution_name} @ file:///app/intergrax"
            source_kind = "path"
        elif norm in closure_agent_ids:
            raise MaterializationLockArtifactLocationBlocked(
                MaterializationLockArtifactLocationBlocked.BLOCKER_CODE
                + f": agent package {package.distribution_name} lacks verified artifact"
            )
        else:
            if package.package_digest is None:
                raise MaterializationLockArtifactIdentityBlocked(
                    MaterializationLockArtifactIdentityBlocked.BLOCKER_CODE
                    + f": package {package.distribution_name} lacks package_digest"
                )
            install_line = (
                f"{package.distribution_name}=={package.version}"
                + _pip_hash_suffix(package.package_digest)
            )

        entries.append(
            LockDrivenInstallEntry(
                distribution_name=package.distribution_name,
                version=package.version,
                install_line=install_line,
                package_digest=package.package_digest,
                source_kind=source_kind,
            )
        )

    manifest_lines = [entry.install_line for entry in entries]
    manifest_text = "\n".join(manifest_lines) + ("\n" if manifest_lines else "")
    return LockDrivenInstallPlan(entries=tuple(entries), manifest_text=manifest_text)


def render_lock_driven_install_manifest(plan: LockDrivenInstallPlan) -> str:
    """Serialize install manifest bytes for staging into build context."""
    return plan.manifest_text


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
    package_artifact_provider: PackageArtifactProvider,
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

    staged_artifacts = stage_lock_authorized_package_artifacts(
        lock=lock,
        candidate_dir=candidate_dir,
        package_artifact_provider=package_artifact_provider,
    )
    validate_lock_package_install_completeness(
        lock=lock,
        build_context=build_context,
        staged_source_roots=tuple(sorted(set(staged))),
        staged_package_artifacts=staged_artifacts,
    )

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
    install_plan = build_lock_driven_install_plan(
        lock=lock,
        build_context=build_context,
        staged_source_roots=tuple(sorted(set(staged))),
        staged_package_artifacts=staged_artifacts,
    )
    (candidate_dir / RUNTIME_INSTALL_MANIFEST_FILENAME).write_text(
        render_lock_driven_install_manifest(install_plan),
        encoding="utf-8",
        newline="\n",
    )
    return tuple(sorted(set(staged)))
