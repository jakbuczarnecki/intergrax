# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness platform ADR path helpers (`docs/project/technical/adr/entries/YYYY-MM-DD/`)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ADR_BASENAME_RE = re.compile(r"^ADR-[A-Z0-9]+-\d+\.md$")
DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ENTRIES_DIRNAME = "entries"


def harness_adr_root(root: Path) -> Path:
    return root / "docs" / "adr"


def harness_adr_entries_dir(adr_root: Path, *, day: date | None = None) -> Path:
    """Directory for new harness ADR files on the given calendar day."""
    d = day or date.today()
    return adr_root / ENTRIES_DIRNAME / d.isoformat()


def harness_adr_entry_path(
    adr_root: Path,
    adr_filename: str,
    *,
    day: date | None = None,
) -> Path:
    """Absolute path for a new harness ADR markdown file."""
    if not ADR_BASENAME_RE.match(adr_filename):
        msg = f"invalid harness ADR filename: {adr_filename!r}"
        raise ValueError(msg)
    return harness_adr_entries_dir(adr_root, day=day) / adr_filename


def harness_adr_entry_relpath(adr_filename: str, *, day: date | None = None) -> str:
    """Repo-relative path under ``docs/project/technical/adr/`` (e.g. ``entries/2026-06-12/ADR-FLOW-001.md``)."""
    d = day or date.today()
    return f"{ENTRIES_DIRNAME}/{d.isoformat()}/{adr_filename}"


def discover_harness_adr_index(adr_root: Path) -> dict[str, str]:
    """Map ADR basename (``ADR-FLOW-001.md``) → day folder (``2026-06-07``)."""
    entries = adr_root / ENTRIES_DIRNAME
    index: dict[str, str] = {}
    if not entries.is_dir():
        return index
    for day_dir in sorted(entries.iterdir()):
        if not day_dir.is_dir() or not DAY_DIR_RE.match(day_dir.name):
            continue
        for path in sorted(day_dir.glob("ADR-*.md")):
            index[path.name] = day_dir.name
    return index


def resolve_harness_adr_relpath(adr_root: Path, adr_basename: str) -> str | None:
    """Return ``entries/<day>/<basename>`` when the ADR exists on disk."""
    day = discover_harness_adr_index(adr_root).get(adr_basename)
    if day is None:
        return None
    return f"{ENTRIES_DIRNAME}/{day}/{adr_basename}"


def relative_harness_adr_link(*, from_day: str, target_basename: str, index: dict[str, str]) -> str:
    """Relative markdown link between harness ADR files under ``entries/``."""
    target_day = index[target_basename]
    if from_day == target_day:
        return target_basename
    return f"../{target_day}/{target_basename}"


def rewrite_harness_adr_cross_links(text: str, *, from_day: str, index: dict[str, str]) -> str:
    """Rewrite same-folder ADR links to correct relative paths after date partitioning."""

    def _replace(match: re.Match[str]) -> str:
        basename = match.group(1)
        if basename not in index:
            return match.group(0)
        link = relative_harness_adr_link(
            from_day=from_day,
            target_basename=basename,
            index=index,
        )
        return f"]({link})"

    return re.sub(r"\]\((ADR-[A-Z0-9]+-\d+\.md)\)", _replace, text)


def deepen_docs_relative_links(text: str) -> str:
    """Adjust ``../`` doc links when an ADR moves from ``docs/project/technical/adr/`` to ``docs/project/technical/adr/entries/<day>/``."""
    replacements = (
        ("](../architecture/", "](../../architecture/"),
        ("](../plan/", "](../../plan/"),
        ("](../intergrax_runtime_architecture.md", "](../../intergrax_runtime_architecture.md"),
        ("](../guides/", "](../../guides/"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text
