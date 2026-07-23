# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical Tier-3 application image builder (runtime-graph isolation).

Materializes a minimal Docker build context from
``applications/<app>/pyproject.toml``, then invokes Docker BuildKit against
that context (never the monorepo root).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from intergrax.applications._shared.application_build_context import (
    materialize_application_build_context,
    render_runtime_graph_manifest,
)
from intergrax.applications._shared.application_runtime_graph import (
    list_application_projects,
    load_application_runtime_graph,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _guess_port(repo_root: Path, application: str) -> int:
    dockerfile = repo_root / "applications" / application / "docker" / "Dockerfile"
    if dockerfile.is_file():
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*EXPOSE\s+(\d+)\s*$", line, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    return 8000


def _guess_env_prefix(application: str) -> str:
    if application.endswith("_application"):
        base = application[: -len("_application")]
    else:
        base = application
    return base.upper() + "_"


def _docker_build(context: Path, *, tag: str) -> int:
    dockerfile = context / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(f"missing Dockerfile in context: {dockerfile}")
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        tag,
        str(context),
    ]
    build_env = os.environ.copy()
    build_env.setdefault("DOCKER_BUILDKIT", "1")
    proc = subprocess.run(cmd, cwd=str(context.parent), env=build_env, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an isolated Tier-3 application image from its declared "
            "runtime graph (minimal materialized context)."
        )
    )
    parser.add_argument(
        "--application",
        required=True,
        help="Application directory name under applications/",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Docker image tag (required unless --manifest-only/--materialize-only)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--context-dir",
        default=None,
        help="Materialize context into this directory instead of a tempfile",
    )
    parser.add_argument(
        "--keep-context",
        action="store_true",
        help="Do not delete the materialized context after the build",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Print sanitized runtime-graph JSON; do not invoke Docker",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Materialize minimal context only; do not invoke Docker",
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--uvicorn-module", default=None)
    parser.add_argument("--env-prefix", default=None)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    application = args.application.strip().strip("/\\")
    known = list_application_projects(repo_root)
    if application not in known:
        print(
            f"RUNTIME_GRAPH_UNRESOLVED: unknown application {application!r}; "
            f"known={known}",
            file=sys.stderr,
        )
        return 2

    graph = load_application_runtime_graph(repo_root, application)
    port = args.port if args.port is not None else _guess_port(repo_root, application)
    env_prefix = args.env_prefix or _guess_env_prefix(application)

    if args.manifest_only and not args.materialize_only:
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
                *[f"agents/{a}/" for a in graph.agent_dirs],
            ],
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0

    if not args.tag and not args.materialize_only:
        print("--tag is required unless --manifest-only/--materialize-only", file=sys.stderr)
        return 2

    keep = bool(args.keep_context or args.context_dir)
    owned_temp: Path | None = None
    if args.context_dir:
        context = Path(args.context_dir)
        if not context.is_absolute():
            context = (repo_root / context).resolve()
    else:
        owned_temp = Path(tempfile.mkdtemp(prefix=f"intergrax-{application}-ctx-"))
        context = owned_temp

    try:
        manifest = materialize_application_build_context(
            repo_root=repo_root,
            application=application,
            output=context,
            pkg_port=port,
            uvicorn_module=args.uvicorn_module,
            env_prefix=env_prefix,
            health_path=args.health_path,
        )
        committed = (
            repo_root / "applications" / application / "docker" / "Dockerfile"
        )
        if committed.is_file():
            df_text = committed.read_text(encoding="utf-8")
            lines = df_text.splitlines()
            # Normalize accidental leading indentation on Docker instructions.
            normalized: list[str] = []
            for ln in lines:
                if ln.startswith("    "):
                    normalized.append(ln[4:])
                else:
                    normalized.append(ln)
            df_text = "\n".join(normalized)
            if committed.read_text(encoding="utf-8").endswith("\n"):
                df_text += "\n"
            (context / "Dockerfile").write_text(df_text, encoding="utf-8", newline="\n")

        print(json.dumps(manifest, sort_keys=True))
        if args.materialize_only:
            return 0
        code = _docker_build(context, tag=args.tag)
        return code
    finally:
        if owned_temp is not None and not keep:
            shutil.rmtree(owned_temp, ignore_errors=True)
        elif owned_temp is not None and keep:
            print(f"kept temporary context: {owned_temp}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
