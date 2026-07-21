# © Artur Czarnecki. All rights reserved.

"""Filesystem policy checks for managed folder sources (LKW-PRODUCT-1)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.tools.providers.filesystem.allowlist import (
    read_allowlist_roots_from_env,
    require_read_allowlist_roots,
    resolve_allowed_path,
)


class SourcePathPolicyError(ValueError):
    """Raised when a candidate source path fails product/filesystem policy."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_under(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def validate_local_folder_source_path(
    path: str,
    *,
    allowlist_roots: frozenset[str] | None = None,
    shadow_roots: tuple[Path, ...] = (),
) -> Path:
    """Validate a local folder source: absolute, allowlisted, readable directory."""
    roots = allowlist_roots if allowlist_roots is not None else read_allowlist_roots_from_env()
    try:
        require_read_allowlist_roots(roots if roots else None)
        resolved = resolve_allowed_path(path, roots)
    except RuntimeError as exc:
        raise SourcePathPolicyError(str(exc)) from exc

    if resolved.is_symlink():
        try:
            resolve_allowed_path(str(resolved.resolve()), roots)
        except RuntimeError as exc:
            raise SourcePathPolicyError("symlink_escapes_allowlist") from exc

    if not resolved.exists():
        raise SourcePathPolicyError("source_path_not_found")
    if not resolved.is_dir():
        raise SourcePathPolicyError("source_path_not_directory")

    for shadow in shadow_roots:
        shadow_resolved = shadow.expanduser().resolve()
        if _is_under(resolved, shadow_resolved) or resolved == shadow_resolved:
            raise SourcePathPolicyError("shadow_workspace_not_allowed_as_source")

    if not os.access(resolved, os.R_OK):
        raise SourcePathPolicyError("source_path_not_readable")

    return resolved
