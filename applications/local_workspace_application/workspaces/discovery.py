# © Artur Czarnecki. All rights reserved.

"""Discover indexable documents under a managed folder source."""

from __future__ import annotations

from pathlib import Path

from intergrax.tools.providers.filesystem.allowlist import resolve_allowed_path

_SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"})


def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED_SUFFIXES


def discover_source_files(
    root: Path,
    *,
    recursive: bool,
    allowlist_roots: frozenset[str],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Return allowlisted supported files and skipped entries (policy failures)."""
    discovered: list[Path] = []
    skipped: list[dict[str, str]] = []
    pattern = "**/*" if recursive else "*"
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_file():
            continue
        if candidate.is_symlink():
            try:
                resolve_allowed_path(str(candidate.resolve()), allowlist_roots)
            except RuntimeError:
                skipped.append({"path": str(candidate), "reason": "symlink_escapes_allowlist"})
                continue
        if not is_supported_document(candidate):
            continue
        try:
            resolved = resolve_allowed_path(str(candidate), allowlist_roots)
        except RuntimeError as exc:
            skipped.append({"path": str(candidate), "reason": str(exc)})
            continue
        if not resolved.is_file():
            skipped.append({"path": str(candidate), "reason": "source_not_found"})
            continue
        discovered.append(resolved)
    return discovered, skipped
