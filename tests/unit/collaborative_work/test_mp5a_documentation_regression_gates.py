# © Artur Czarnecki. All rights reserved.

"""MP-5A — documentation regression gates for ContextView ownership freeze."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_COLLAB_ARCH = _REPO_ROOT / "docs" / "project" / "architecture" / "COLLABORATIVE_WORK.md"
_COLLAB_PLAN = _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "COLLABORATIVE_WORK.md"
_MP_ARCH = _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
_MP_PLAN = _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md"
_ADR_MP006 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-17"
    / "ADR-MP-006.md"
)

_STATUS_DOCS: dict[str, Path] = {
    "collaborative_work_architecture": _COLLAB_ARCH,
    "collaborative_work_plan": _COLLAB_PLAN,
    "multiplayer_architecture": _MP_ARCH,
    "multiplayer_plan": _MP_PLAN,
}

_REQUIRED_MARKERS = (
    "MP-5A — APPROVED / CLOSED",
    "MP-5 ownership — FROZEN",
    "MP-5B — NEXT",
    "ADR-MP-006",
)

_ANTI_SUBSTITUTION = (
    "UCL ≠ Principal-scoped ContextView",
    "Memory ≠ Principal-scoped ContextView",
    "Context Engineering ≠ Principal-scoped ContextView",
    "ContextView ≠ storage",
)

_FORBIDDEN = (
    re.compile(r"OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION", re.I),
    re.compile(r"MP-5 view composition store", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp5a_adr_accepted_on_disk() -> None:
    text = _read(_ADR_MP006)
    assert "Accepted" in text
    assert "ContextView ≠ storage" in text


def test_mp5a_status_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [m for m in _REQUIRED_MARKERS if m not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_mp5a_anti_substitution_in_multiplayer_architecture() -> None:
    text = _read(_MP_ARCH)
    missing = [m for m in _ANTI_SUBSTITUTION if m not in text]
    assert not missing, f"multiplayer_architecture: missing anti-substitution: {missing}"


def test_mp5a_no_forbidden_ownership_drift_in_mp5_section() -> None:
    text = _read(_MP_ARCH)
    mp5_start = text.find("### MP-5")
    assert mp5_start >= 0, "MP-5 section missing"
    mp6_start = text.find("### MP-6", mp5_start)
    mp5_block = text[mp5_start:mp6_start] if mp6_start > mp5_start else text[mp5_start : mp5_start + 2500]
    for pattern in _FORBIDDEN:
        assert pattern.search(mp5_block) is None, (
            f"forbidden drift in MP-5 block: {pattern.pattern}"
        )
