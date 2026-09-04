"""Safe relative path handling for proof data packages."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from intergrax.proof_data.errors import DataPackageDescriptorError

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")


def normalize_relative_path(relative_path: str) -> str:
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        raise DataPackageDescriptorError("relative_path must be non-empty")
    if normalized.startswith("/"):
        raise DataPackageDescriptorError("relative_path must be relative")
    if _WINDOWS_ABSOLUTE.match(normalized):
        raise DataPackageDescriptorError("relative_path must not be an absolute Windows path")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise DataPackageDescriptorError("relative_path must not contain parent traversal")
    return normalized


def resolve_under_root(root: Path, relative_path: str) -> Path:
    normalized = normalize_relative_path(relative_path)
    root_resolved = root.resolve()
    candidate = (root_resolved / Path(*PurePosixPath(normalized).parts)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise DataPackageDescriptorError(
            f"relative_path escapes installation root: {relative_path}"
        ) from exc
    return candidate
