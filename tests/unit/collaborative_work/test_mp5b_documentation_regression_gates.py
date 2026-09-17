# © Artur Czarnecki. All rights reserved.

"""MP-5B — documentation regression gates for ContextView contracts closure."""

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
    "MP-5B — APPROVED / CLOSED",
    "MP-5 ownership — FROZEN",
    "intergrax/contracts/context_view.py",
)

_CONTRACT_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "context_view.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp5b_contract_module_exists() -> None:
    assert _CONTRACT_MODULE.is_file()


def test_mp5b_status_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [marker for marker in _REQUIRED_MARKERS if marker not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_mp5b_no_context_view_storage_class_name() -> None:
    text = _read(_CONTRACT_MODULE)
    assert "ContextViewDatabase" not in text
    assert "ContextViewStorage" not in text
