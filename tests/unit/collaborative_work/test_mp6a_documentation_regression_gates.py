# © Artur Czarnecki. All rights reserved.

"""MP-6A — documentation regression gates for Activity & Provenance ownership freeze."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_STATUS_DOCS = {
    "collaborative_work_architecture": _REPO_ROOT
    / "docs"
    / "project"
    / "architecture"
    / "COLLABORATIVE_WORK.md",
    "collaborative_work_plan": _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "plans"
    / "COLLABORATIVE_WORK.md",
    "multiplayer_architecture": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "architecture"
    / "MULTIPLAYER_AI.md",
    "multiplayer_plan": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "plan"
    / "MULTIPLAYER_AI.md",
}

_ADR_MP007 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-18"
    / "ADR-MP-007.md"
)

_REQUIRED_MARKERS = (
    "MP-6A — APPROVED / CLOSED",
    "MP-6 ownership — FROZEN",
    "ADR-MP-007",
    "intergrax/contracts/collaborative_activity.py",
)

_ANTI_SUBSTITUTION = (
    "Collaborative Activity != Runtime Trace",
    "Activity ≠ observability",
)

_FORBIDDEN = (
    re.compile(r"OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp6a_adr_accepted_on_disk() -> None:
    text = _read(_ADR_MP007)
    assert "Accepted" in text
    assert "Collaborative Activity != Runtime Trace" in text or "Activity ≠" in text


def test_mp6a_status_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [marker for marker in _REQUIRED_MARKERS if marker not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_mp6a_anti_substitution_in_multiplayer_architecture() -> None:
    text = _read(_STATUS_DOCS["multiplayer_architecture"])
    missing = [m for m in _ANTI_SUBSTITUTION if m not in text]
    assert not missing, f"multiplayer_architecture: missing anti-substitution: {missing}"


def test_mp6a_no_forbidden_ownership_drift_in_mp6_section() -> None:
    text = _read(_STATUS_DOCS["multiplayer_architecture"])
    mp6_start = text.find("### MP-6")
    assert mp6_start >= 0, "MP-6 section missing"
    mp7_start = text.find("### MP-7", mp6_start)
    mp6_block = text[mp6_start:mp7_start] if mp7_start > mp6_start else text[mp6_start : mp6_start + 4000]
    for pattern in _FORBIDDEN:
        assert pattern.search(mp6_block) is None, (
            f"forbidden drift in MP-6 block: {pattern.pattern}"
        )
