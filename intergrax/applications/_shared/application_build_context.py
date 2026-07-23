# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Materialize a minimal Docker build context from an application runtime graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

from intergrax.applications._shared.application_runtime_graph import (
    ApplicationRuntimeGraph,
    load_application_runtime_graph,
    list_application_projects,
)
from intergrax.applications._shared.docker_templates import render_runtime_graph_dockerfile

SCHEMA_VERSION = 2

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
    re.compile(
        r"(?i)\b(?:[A-Z0-9_]+(?:TOKEN|API_KEY|SECRET|PASSWORD))\s*=\s*(?!$|"
        r"(?:changeme|placeholder|your[_-]?|xxx|<|\.\.\.|\$\{))[^\s\"']{8,}"
    ),
)

_PLACEHOLDER_OK = re.compile(
    r"(?i)(changeme|placeholder|your[_-]|xxx|redacted|example|dummy|\$\{|<.+>)"
)


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
            raise ValueError(f"symlink escape rejected: {path} -> {target}")


def _should_skip(path: Path, *, repo_root: Path) -> bool:
    rel_parts = path.relative_to(repo_root).parts if _is_within(repo_root, path) else path.parts
    if any(part in _SKIP_DIR_NAMES for part in rel_parts):
        return True
    if path.name in _SKIP_FILE_NAMES:
        return True
    if path.suffix in _SKIP_SUFFIXES:
        return True
    if path.name.endswith(".egg-info"):
        return True
    return False


def scan_secrets(path: Path, content: bytes) -> None:
    """Fail closed when non-placeholder secrets are present in build context files."""
    # Scan only credential-bearing / config surfaces — not general Python source
    # (which routinely mentions *_SECRET env var *names*).
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
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".py"}:
        # Still scan .py for high-signal token prefixes only.
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return
        for pattern in _SECRET_PATTERNS[:3]:  # xapp-, xoxb-, Bearer
            if pattern.search(text) and not _PLACEHOLDER_OK.search(text):
                raise ValueError(
                    f"secret-like content rejected in build context: {path}"
                )
        return
    if name_l not in sensitive_names and suffix not in {
        ".env",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    } and not name_l.endswith(".env"):
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
            if path.name.endswith(".example") and "=" in snippet:
                _key, _, value = snippet.partition("=")
                if not value.strip() or _PLACEHOLDER_OK.search(value):
                    continue
            raise ValueError(
                f"secret-like content rejected in build context: {path} ({pattern.pattern})"
            )


def _copy_filtered_tree(
    src: Path,
    dst: Path,
    *,
    repo_root: Path,
    extra_skip_dirs: frozenset[str] | None = None,
) -> list[str]:
    """Copy ``src`` → ``dst`` honoring skip rules. Returns relative POSIX paths copied."""
    included: list[str] = []
    skip_dirs = _SKIP_DIR_NAMES | (extra_skip_dirs or frozenset())
    if not src.exists():
        raise FileNotFoundError(src)
    _reject_symlink_escape(repo_root, src)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        current = Path(dirpath)
        _reject_symlink_escape(repo_root, current)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in skip_dirs and not d.endswith(".egg-info")
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


def _rewrite_workspace_members(root_pyproject: str, members: list[str]) -> str:
    start = root_pyproject.index("[tool.uv.workspace]")
    # members table ends before next top-level TOML table at column 0 that is not comments
    rest = root_pyproject[start:]
    # Find end: next [[ or [ that starts a new section after members
    match = re.search(
        r"\[tool\.uv\.workspace\][^\[]*?members\s*=\s*\[[^\]]*\]",
        root_pyproject,
        flags=re.S,
    )
    if not match:
        raise ValueError("RUNTIME_GRAPH_UNRESOLVED: cannot locate workspace members")
    member_lines = ",\n".join(f'  "{m}"' for m in members)
    replacement = (
        "[tool.uv.workspace]\n"
        "# Materialized subset — selected application runtime graph only.\n"
        f"members = [\n{member_lines},\n]\n"
    )
    return root_pyproject[: match.start()] + replacement + root_pyproject[match.end() :]


def _lock_digest(lock_path: Path) -> str:
    digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return digest


def render_runtime_graph_manifest(
    graph: ApplicationRuntimeGraph,
    *,
    repo_root: Path,
    excluded_applications: list[str],
    included_source_roots: list[str],
) -> dict[str, object]:
    """Serialize the runtime graph for ``.intergrax-runtime-graph.json``.

    ``direct_third_party_distributions`` lists only third-party dependencies
    declared by the Tier-3 application. Agent-declared third-party packages are
    resolved by uv through selected agent projects / ``uv.lock`` — not flattened
    into this field.
    """
    lock_path = repo_root / "uv.lock"
    return {
        "schema_version": SCHEMA_VERSION,
        "application": graph.application,
        "application_distribution": graph.application_dist,
        "platform_packages": ["Intergrax-ai"],
        "platform_extras": list(graph.platform_extras),
        "direct_agent_packages": list(graph.direct_agent_distributions),
        "transitive_agent_packages": list(graph.transitive_agent_distributions),
        "all_agent_packages": list(graph.all_agent_distributions),
        "direct_agent_dirs": list(graph.direct_agent_dirs),
        "transitive_agent_dirs": list(graph.transitive_agent_dirs),
        "all_agent_dirs": list(graph.all_agent_dirs),
        "direct_third_party_distributions": list(
            graph.direct_third_party_distributions
        ),
        "included_source_roots": included_source_roots,
        "excluded_tier3_applications": excluded_applications,
        "lock_digest": _lock_digest(lock_path) if lock_path.is_file() else "",
    }


def materialize_application_build_context(
    *,
    repo_root: Path,
    application: str,
    output: Path,
    pkg_port: int | None = None,
    uvicorn_module: str | None = None,
    env_prefix: str | None = None,
    health_path: str = "/health",
) -> dict[str, object]:
    """Create a minimal auditable build context for one application image."""
    repo_root = repo_root.resolve()
    output = output.resolve()
    if not _is_within(repo_root, output) and output.exists():
        # Allow temp dirs outside repo (canonical CLI uses tempfile).
        pass
    if output.exists():
        if output == repo_root:
            raise ValueError("refusing to materialize into repository root")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    graph = load_application_runtime_graph(repo_root, application)
    all_apps = list_application_projects(repo_root)
    excluded_apps = [a for a in all_apps if a != graph.application]

    included_roots: list[str] = []

    # Root metadata
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        src = repo_root / name
        if not src.is_file():
            raise FileNotFoundError(src)
        data = src.read_bytes()
        scan_secrets(src, data)
        if name == "pyproject.toml":
            text = data.decode("utf-8")
            text = _rewrite_workspace_members(text, list(graph.workspace_member_paths))
            (output / name).write_text(text, encoding="utf-8", newline="\n")
        else:
            (output / name).write_bytes(data)
        included_roots.append(name)

    # Platform source
    _copy_filtered_tree(
        repo_root / "intergrax",
        output / "intergrax",
        repo_root=repo_root,
        extra_skip_dirs=frozenset({"tests"}),
    )
    included_roots.append("intergrax/")

    # Selected application only (no sibling Tier-3 trees)
    apps_out = output / "applications"
    apps_out.mkdir(parents=True, exist_ok=True)
    (apps_out / "__init__.py").write_text(
        "# © Artur Czarnecki. All rights reserved.\n",
        encoding="utf-8",
        newline="\n",
    )
    _copy_filtered_tree(
        repo_root / "applications" / graph.application,
        apps_out / graph.application,
        repo_root=repo_root,
        extra_skip_dirs=frozenset(
            {
                "docs",
                "tests",
                "notebooks",
                "proof",
                "docker",
            }
        ),
    )
    included_roots.append(f"applications/{graph.application}/")

    # All reachable agents (direct + transitive); agents/ always present for COPY
    agents_out = output / "agents"
    agents_out.mkdir(parents=True, exist_ok=True)
    (agents_out / ".keep").write_text("", encoding="utf-8", newline="\n")
    for agent in graph.all_agent_dirs:
        _copy_filtered_tree(
            repo_root / "agents" / agent,
            agents_out / agent,
            repo_root=repo_root,
            extra_skip_dirs=frozenset({"docs", "tests", "notebooks", "prompts"}),
        )
        included_roots.append(f"agents/{agent}/")

    # Generic Dockerfile for materialized context
    dockerfile = render_runtime_graph_dockerfile(
        pkg=graph.application,
        port=pkg_port or 8000,
        env_prefix=env_prefix or _guess_env_prefix(graph.application),
        health_path=health_path,
        uvicorn_module=uvicorn_module,
    )
    (output / "Dockerfile").write_text(dockerfile, encoding="utf-8", newline="\n")
    included_roots.append("Dockerfile")

    # Protective dockerignore inside context (defense in depth)
    (output / ".dockerignore").write_text(
        "\n".join(
            [
                ".git",
                ".venv",
                "**/.venv",
                ".env",
                "**/.env",
                "**/__pycache__",
                "**/.pytest_cache",
                "**/.mypy_cache",
                "**/.ruff_cache",
                "**/proof",
                "**/docs",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )

    manifest = render_runtime_graph_manifest(
        graph,
        repo_root=repo_root,
        excluded_applications=excluded_apps,
        included_source_roots=sorted(included_roots),
    )
    (output / ".intergrax-runtime-graph.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Fail closed: no other Tier-3 application directories
    for child in (output / "applications").iterdir():
        if child.name in {"__init__.py", graph.application}:
            continue
        raise ValueError(
            f"DOCKER_ISOLATION_FAILED: unexpected path in context: {child}"
        )

    # Fail closed: actual agent directories == expected all_agent_dirs
    # (ignore intentional sentinel files such as .keep)
    actual_agent_dirs = sorted(
        p.name for p in agents_out.iterdir() if p.is_dir()
    )
    expected_agent_dirs = sorted(graph.all_agent_dirs)
    if actual_agent_dirs != expected_agent_dirs:
        raise ValueError(
            "DOCKER_ISOLATION_FAILED: agent directory mismatch in context: "
            f"actual={actual_agent_dirs} expected={expected_agent_dirs}"
        )

    return manifest


def _guess_env_prefix(application: str) -> str:
    if application.endswith("_application"):
        base = application[: -len("_application")]
    else:
        base = application
    return base.upper() + "_"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a minimal application Docker build context"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Application project path, e.g. applications/local_workspace_application",
    )
    parser.add_argument("--output", required=True, help="Output directory for context")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--uvicorn-module", default=None)
    parser.add_argument("--env-prefix", default=None)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print sanitized runtime graph JSON without writing a full context",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    project = args.project.replace("\\", "/").strip("/")
    if not project.startswith("applications/"):
        application = project
    else:
        application = project.split("/", 1)[1]

    if args.manifest_only:
        graph = load_application_runtime_graph(repo_root, application)
        all_apps = list_application_projects(repo_root)
        manifest = render_runtime_graph_manifest(
            graph,
            repo_root=repo_root,
            excluded_applications=[a for a in all_apps if a != application],
            included_source_roots=[
                "pyproject.toml",
                "uv.lock",
                "README.md",
                "intergrax/",
                f"applications/{application}/",
                *[f"agents/{a}/" for a in graph.all_agent_dirs],
            ],
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    manifest = materialize_application_build_context(
        repo_root=repo_root,
        application=application,
        output=Path(args.output),
        pkg_port=args.port,
        uvicorn_module=args.uvicorn_module,
        env_prefix=args.env_prefix,
        health_path=args.health_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
