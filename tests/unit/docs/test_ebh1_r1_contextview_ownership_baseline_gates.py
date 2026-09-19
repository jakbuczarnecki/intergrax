# © Artur Czarnecki. All rights reserved.

"""EBH-1-R1 — documentation regression gates for ContextView ownership baseline."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EBH1 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "audits"
    / "EBH-1_CANONICAL_BOUNDARY_BASELINE_RECONCILIATION.md"
)
_EBH1_R1 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "audits"
    / "EBH-1-R1_CONTEXTVIEW_OWNERSHIP_BASELINE_CORRECTION.md"
)

_REQUIRED_ACTIVE = (
    "CONTEXT_ASSEMBLY_AUTHORITY",
    "PRINCIPAL_SCOPED_CONTEXT_VIEW_AUTHORITY",
    "COLLABORATIVE_WORK (MP-5)",
    "EBH-F-H-004",
    "RESOLVED BY EBH-1-R1",
)

_FORBIDDEN_ACTIVE = re.compile(
    r"\|\s*CONTEXT_VIEW_COMPOSITION_AUTHORITY\s*\|\s*CONTEXT_ENGINEERING",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_ebh1_r1_required_markers_in_baseline() -> None:
    text = _read(_EBH1)
    missing = [m for m in _REQUIRED_ACTIVE if m not in text]
    assert not missing, f"EBH-1 baseline missing: {missing}"


def test_ebh1_r1_forbids_active_context_view_composition_authority_row() -> None:
    text = _read(_EBH1)
    assert not _FORBIDDEN_ACTIVE.search(text), (
        "EBH-1 must not list active CONTEXT_VIEW_COMPOSITION_AUTHORITY → CONTEXT_ENGINEERING"
    )


def test_ebh1_r1_record_exists_and_references_adr() -> None:
    text = _read(_EBH1_R1)
    assert "ADR-MP-006" in text
    assert "CLOSED / CERTIFIED" in text
