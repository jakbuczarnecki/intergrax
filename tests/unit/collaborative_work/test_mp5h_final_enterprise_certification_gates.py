# © Artur Czarnecki. All rights reserved.

"""MP-5H — final cross-slice enterprise certification gates (ContextView / MP-5)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]

_QUALIFICATION = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-5H_FINAL_ENTERPRISE_CERTIFICATION.md"
)
_COLLAB_ARCH = _REPO / "docs" / "project" / "architecture" / "COLLABORATIVE_WORK.md"
_COLLAB_PLAN = _REPO / "docs" / "project" / "maintainers" / "plans" / "COLLABORATIVE_WORK.md"
_MP_ARCH = _REPO / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
_MP_PLAN = _REPO / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md"

_COMPOSER = _REPO / "intergrax" / "collaborative_work" / "context_view_composition.py"
_VISIBILITY = _REPO / "intergrax" / "collaborative_work" / "context_view_visibility.py"
_WIRING = _REPO / "intergrax" / "collaborative_work" / "context_view_source_wiring.py"
_ADAPTERS = _REPO / "intergrax" / "collaborative_work" / "context_view_source_adapters.py"

_STATUS_DOCS: dict[str, Path] = {
    "collaborative_work_architecture": _COLLAB_ARCH,
    "collaborative_work_plan": _COLLAB_PLAN,
    "multiplayer_architecture": _MP_ARCH,
    "multiplayer_plan": _MP_PLAN,
}

_REQUIRED_CLOSURE_MARKERS = (
    "MP-5 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-5H — CLOSED / FINAL CERTIFICATION PASSED",
    "MP-6 — NEXT",
)

_CAPABILITY_STATEMENT = (
    "ContextView is a principal-scoped, reference-first, "
    "policy-governed projection over source-domain canonical references."
)

_FORBIDDEN_ACTIVE_DRIFT = (
    re.compile(r"\*\*MP-5H\s+—\s+NEXT\*\*", re.I),
    re.compile(r"Current active task:\*\*\s+MP-5G\s+—\s+NEXT", re.I),
    re.compile(r"Next task:\*\*\s+\*\*MP-5G\s+—\s+NEXT", re.I),
)

_PRODUCTION_FLOW_MODULES = (
    _VISIBILITY,
    _COMPOSER,
    _WIRING,
    _ADAPTERS,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_mp5h_qualification_record_exists_and_passed() -> None:
    text = _read(_QUALIFICATION)
    assert "MP-5H — FINAL ENTERPRISE CERTIFICATION PASSED" in text
    assert "MP-5 — ENTERPRISE CERTIFIED / CLOSED" in text
    assert "BLOCKING FINDINGS: NONE" in text


def test_mp5h_closure_markers_present_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        missing = [marker for marker in _REQUIRED_CLOSURE_MARKERS if marker not in text]
        assert not missing, f"{name}: missing MP-5H closure markers: {missing}"


def test_mp5h_capability_statement_in_collaborative_work_architecture() -> None:
    text = _read(_COLLAB_ARCH)
    assert _CAPABILITY_STATEMENT in text


def test_mp5h_no_stale_active_task_drift_in_ssot_docs() -> None:
    for name, path in _STATUS_DOCS.items():
        text = _read(path)
        for pattern in _FORBIDDEN_ACTIVE_DRIFT:
            match = pattern.search(text)
            assert match is None, f"{name}: stale active-task drift: {match.group(0)!r}"


def test_mp5h_production_flow_modules_exist() -> None:
    missing = [path for path in _PRODUCTION_FLOW_MODULES if not path.is_file()]
    assert not missing, missing


def test_mp5h_composer_has_no_default_adapter_or_wiring_imports() -> None:
    text = _read(_COMPOSER)
    for symbol in (
        "context_view_source_adapters",
        "context_view_source_wiring",
        "DefaultMemoryContextSource",
        "hydrate",
    ):
        assert symbol not in text


def test_mp5h_adapters_have_no_asyncio_run() -> None:
    text = _read(_ADAPTERS)
    assert "asyncio.run" not in text


def test_mp5h_wiring_asyncio_run_is_composition_root_only() -> None:
    tree = ast.parse(_read(_WIRING), filename=str(_WIRING))
    adapter_text = _read(_ADAPTERS)
    assert "asyncio.run" in _read(_WIRING)
    assert "asyncio.run" not in adapter_text


def test_mp5h_visibility_evaluator_does_not_import_composer() -> None:
    tree = ast.parse(_read(_VISIBILITY), filename=str(_VISIBILITY))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any("context_view_composition" in module for module in modules)
