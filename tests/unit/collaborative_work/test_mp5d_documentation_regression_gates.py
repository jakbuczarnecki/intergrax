# © Artur Czarnecki. All rights reserved.

"""MP-5D — documentation regression gates for source composition ports closure."""

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
    "MP-5D — APPROVED / CLOSED",
    "MP-5E — APPROVED / CLOSED",
    "MP-5F — ENTERPRISE SOURCE INTEGRATION CERTIFIED / CLOSED",
    "intergrax/contracts/context_view_source_ports.py",
    "ContextViewPolicyDecision",
    "source composition ports",
)

_CONTRACT_MODULE = _REPO_ROOT / "intergrax" / "contracts" / "context_view_source_ports.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp5d_source_ports_contract_module_exists() -> None:
    assert _CONTRACT_MODULE.is_file()


def test_mp5d_status_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        text_lower = text.lower()
        missing = [
            marker
            for marker in _REQUIRED_MARKERS
            if marker.lower() not in text_lower and marker not in text
        ]
        assert not missing, f"{name}: missing markers: {missing}"


def test_mp5d_no_default_composer_or_adapter_symbols_in_contract() -> None:
    text = _read(_CONTRACT_MODULE)
    forbidden = (
        "DefaultMemoryContextSource",
        "ContextViewComposer",
        "ContextViewDatabase",
    )
    for symbol in forbidden:
        assert symbol not in text, symbol
