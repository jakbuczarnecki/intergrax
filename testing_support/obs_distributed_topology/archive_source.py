# © Artur Czarnecki. All rights reserved.

"""Immutable git-archive source tree for exact-SHA child processes."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path


def materialize_git_archive_source_tree(
    repo_root: Path,
    sha: str,
    work_dir: Path,
) -> Path:
    """
    Extract ``git archive`` for ``sha`` into ``work_dir/src``.

    Returns the directory that must be prepended to ``PYTHONPATH`` / ``sys.path``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_tar = work_dir / "qualification-tree.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", sha, "-o", str(archive_tar)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    extract_root = work_dir / "src"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_tar, "r") as tar:
        tar.extractall(path=extract_root, filter="data")
    return extract_root.resolve()


def resolve_intergrax_import_root(source_root: Path) -> Path:
    """Return the path whose children include the ``intergrax`` package."""
    resolved = source_root.resolve()
    if (resolved / "intergrax").is_dir():
        return resolved
    nested = resolved / "intergrax"
    if nested.is_dir() and (nested / "__init__.py").is_file():
        return resolved
    raise FileNotFoundError(f"intergrax package not found under archive root {resolved}")
