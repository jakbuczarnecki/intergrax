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
    "MP-6A — CLOSED / RECERTIFIED",
    "MP-6A-C1",
    "MP-6 ownership — FROZEN",
    "ADR-MP-007",
    "intergrax/contracts/collaborative_activity.py",
    "CollaborativeActivityTypeId",
    "opaque",
)

_ANTI_SUBSTITUTION = (
    "Collaborative Activity != Runtime Trace",
    "Activity ≠ observability",
)

_FORBIDDEN = (
    re.compile(r"OWNERSHIP_TO_CONFIRM_BEFORE_IMPLEMENTATION", re.I),
)

_FORBIDDEN_MP6_ORDERING = re.compile(
    r"cursor timeline sorted by `\s*\(occurred_at,\s*activity_id\)`",
    re.I,
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


def test_mp6a_c1_no_occurred_at_only_pagination_claim_in_collaborative_work() -> None:
    text = _read(_STATUS_DOCS["collaborative_work_architecture"])
    mp6_start = text.find("## Collaborative Activity & Provenance")
    assert mp6_start >= 0
    mp6_block = text[mp6_start : mp6_start + 6000]
    assert _FORBIDDEN_MP6_ORDERING.search(mp6_block) is None


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


def test_mp6e_closure_current_status_points_to_mp6f_next() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        assert "**MP-6E — NEXT**" not in text, f"{name}: stale active MP-6E NEXT"
        assert "MP-6F" in text, f"{name}: missing MP-6F roadmap"
    adr = _read(_ADR_MP007)
    assert "MP-6E - NEXT" not in adr, "ADR still marks MP-6E NEXT"
    assert "MP-6F" in adr, "ADR missing MP-6F roadmap"
