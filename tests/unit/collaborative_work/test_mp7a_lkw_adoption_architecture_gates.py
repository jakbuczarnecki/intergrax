# © Artur Czarnecki. All rights reserved.

"""MP-7A — LKW Multiplayer adoption architecture and ADR-MP-008 gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

_LKW_APP = _REPO_ROOT / "applications" / "local_workspace_application"

_ADR_MP008 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-19"
    / "ADR-MP-008.md"
)

_GATE_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7A_LKW_MULTIPLAYER_ADOPTION_ARCHITECTURE_GATE.md"
)

_STATUS_DOCS = {
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
    "lkw_architecture": _LKW_APP / "docs" / "ARCHITECTURE.md",
    "lkw_plan": _LKW_APP / "docs" / "IMPLEMENTATION_PLAN.md",
}

_REQUIRED_MARKERS = (
    "MP-7A",
    "ADR-MP-008",
    "MP-INV-30",
    "MP-7B",
    "reference consumer",
)

_FORBIDDEN_LKW_IMPORT_PREFIXES = (
    "intergrax.collaborative_work",
    "intergrax.collaborative_work.",
)

_FORBIDDEN_IDENTITY_UTTERANCES = (
    re.compile(r"channel_id\s*==\s*workspace_id", re.I),
    re.compile(r"Conversation Context\s*==\s*ContextView", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _lkw_python_files() -> list[Path]:
    if not _LKW_APP.is_dir():
        return []
    skip = {"tests", "docker"}
    return [p for p in _LKW_APP.rglob("*.py") if not any(part in skip for part in p.parts)]


def _imports_in_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def test_mp7a_adr_accepted_on_disk() -> None:
    text = _read(_ADR_MP008)
    assert "Accepted" in text
    assert "reference consumer" in text.lower()
    assert "does not transfer ownership" in text.lower() or "not transfer ownership" in text.lower()


def test_mp7a_gate_doc_closed() -> None:
    text = _read(_GATE_DOC)
    assert "CLOSED / CERTIFIED" in text
    assert "Option B" in text or "collaborative_workspace_ref" in text
    assert "BLOCKING ARCHITECTURE GAPS: NONE" in text


def test_mp7a_status_markers_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [m for m in _REQUIRED_MARKERS if m not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_mp7a_no_forbidden_identity_collapse_in_gate_doc() -> None:
    text = _read(_GATE_DOC)
    for pattern in _FORBIDDEN_IDENTITY_UTTERANCES:
        assert pattern.search(text) is None, f"forbidden identity collapse: {pattern.pattern}"


def test_mp7a_lkw_no_collaborative_work_implementation_imports() -> None:
    violations: list[str] = []
    for path in _lkw_python_files():
        for mod in _imports_in_file(path):
            for prefix in _FORBIDDEN_LKW_IMPORT_PREFIXES:
                if mod == prefix.rstrip(".") or mod.startswith(prefix):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}: {mod}")
    assert not violations, "LKW must not import collaborative_work implementation: " + ", ".join(
        violations
    )


def test_mp7a_adr_register_lists_mp008() -> None:
    readme = _read(_REPO_ROOT / "docs" / "project" / "technical" / "adr" / "README.md")
    assert "ADR-MP-008" in readme
    row = next(line for line in readme.splitlines() if "ADR-MP-008" in line)
    assert "Accepted" in row
