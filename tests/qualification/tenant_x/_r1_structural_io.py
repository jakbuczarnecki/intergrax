# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 bounded production text checks (no full-repo scan)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]


def read_production(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def structural_patterns_present(rel: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    text = read_production(rel)
    return tuple(p for p in patterns if p in text)


def all_structural_patterns_present(
    checks: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Each check is (relative_path, pattern). Returns matched pattern descriptions."""
    matched: list[str] = []
    for rel, pattern in checks:
        if pattern not in read_production(rel):
            return ()
        matched.append(f"{rel}::{pattern}")
    return tuple(matched)


def discovery_markers(rel: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    """Candidate-discovery hints only — not continuity proof."""
    return structural_patterns_present(rel, markers)
