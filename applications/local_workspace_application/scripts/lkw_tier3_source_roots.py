#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Canonical Tier-3 host-side source import roots for LKW public proof launchers.

Tier-3 application projects use ``[tool.uv] package = false``; application modules
remain importable via ``PYTHONPATH=applications`` (see APPLICATION_DEPENDENCY_MODEL).
Host-side proof workloads that import ``local_workspace_application`` must establish
this context before ``uv run --project applications/<app>``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def resolve_tier3_source_roots(repo_root: Path) -> tuple[Path, Path]:
    """Return canonical ``applications`` and ``agents`` source roots for a repository."""
    root = repo_root.resolve()
    applications = root / "applications"
    agents = root / "agents"
    if not applications.is_dir():
        raise ValueError(f"missing applications source root: {applications}")
    if not agents.is_dir():
        raise ValueError(f"missing agents source root: {agents}")
    return applications, agents


def format_windows_path_list(
    repo_root: Path,
    *,
    existing: str | None = None,
) -> str:
    """Render ``PYTHONPATH`` for Windows launchers; preserve any existing entries."""
    applications, agents = resolve_tier3_source_roots(repo_root)
    prefix = f"{applications}{os.pathsep}{agents}"
    merged_existing = existing if existing is not None else os.environ.get("PYTHONPATH")
    if merged_existing:
        return f"{prefix}{os.pathsep}{merged_existing}"
    return prefix


def ensure_tier3_source_roots_on_sys_path(repo_root: Path) -> tuple[Path, Path]:
    """Insert canonical source roots on ``sys.path`` when invoked from Python entrypoints."""
    applications, agents = resolve_tier3_source_roots(repo_root)
    for path in (applications, agents):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return applications, agents


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve canonical Tier-3 source import roots for LKW proof launchers.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repository root (must contain applications/ and agents/).",
    )
    parser.add_argument(
        "--format",
        choices=("windows-path-list",),
        required=True,
        help="Output format for launcher consumption.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.format == "windows-path-list":
        print(
            format_windows_path_list(
                args.repo_root,
                existing=os.environ.get("PYTHONPATH"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
