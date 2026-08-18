# © Artur Czarnecki. All rights reserved.
"""Shared domain-id helpers for documentation integrity checks."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB = REPO_ROOT / "docs" / "project" / "architecture" / "intergrax_runtime_architecture.md"

_HUB_DOMAIN_ROW = re.compile(
    r"^\| \d+ \| `([A-Z][A-Z0-9_]+)` \| \[`\1\.md`\]",
    re.MULTILINE,
)


def canonical_domain_ids() -> tuple[str, ...]:
    """Return canonical runtime domain ids in hub table order."""
    ids = _HUB_DOMAIN_ROW.findall(HUB.read_text(encoding="utf-8"))
    if not ids:
        msg = "hub domain index is empty"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"duplicate hub domain ids: {ids}"
        raise ValueError(msg)
    return tuple(ids)


DOMAIN_ORDER: tuple[str, ...] = canonical_domain_ids()
