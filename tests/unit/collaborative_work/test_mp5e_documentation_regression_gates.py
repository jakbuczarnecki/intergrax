# © Artur Czarnecki. All rights reserved.

"""MP-5E — documentation regression gates for default composer closure."""

from __future__ import annotations

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

_REQUIRED_MARKERS = (
    "MP-5E — APPROVED / CLOSED",
    "MP-5F — ENTERPRISE SOURCE INTEGRATION CERTIFIED / CLOSED",
    "context_view_source_adapters",
    "intergrax/contracts/context_view_composition.py",
    "DefaultContextViewComposer",
    "ContextViewComposer",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp5e_status_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        text_lower = text.lower()
        missing = [
            marker
            for marker in _REQUIRED_MARKERS
            if marker.lower() not in text_lower and marker not in text
        ]
        assert not missing, f"{name}: missing markers: {missing}"
