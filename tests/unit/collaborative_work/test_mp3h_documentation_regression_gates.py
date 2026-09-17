# © Artur Czarnecki. All rights reserved.

"""MP-3H — documentation regression gates for MP-3 enterprise closure status."""

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

_STATUS_DOCS: dict[str, Path] = {
    "collaborative_work_architecture": _COLLAB_ARCH,
    "collaborative_work_plan": _COLLAB_PLAN,
    "multiplayer_architecture": _MP_ARCH,
    "multiplayer_plan": _MP_PLAN,
}

_REQUIRED_STATUS_MARKERS = (
    "MP-3 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-3A…MP-3H — APPROVED / CLOSED",
    "MP-5B — NEXT",
    "MP-5 ownership — FROZEN",
)

_FORBIDDEN_ACTIVE_DRIFT = (
    re.compile(r"MP-3H\s+—\s+NOT\s+STARTED", re.I),
    re.compile(r"MP-3\s+runtime\s+implementation\s+\*\*IN\s+PROGRESS\*\*", re.I),
    re.compile(r"MP-3\s+runtime\s+\*\*IN\s+PROGRESS\*\*", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp3h_status_markers_present_in_all_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [marker for marker in _REQUIRED_STATUS_MARKERS if marker not in text]
        assert not missing, f"{name}: missing closure markers: {missing}"


def test_mp3h_no_active_runtime_in_progress_drift() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        for pattern in _FORBIDDEN_ACTIVE_DRIFT:
            match = pattern.search(text)
            assert match is None, (
                f"{name}: forbidden active MP-3 drift: {match.group(0)!r}"
            )
