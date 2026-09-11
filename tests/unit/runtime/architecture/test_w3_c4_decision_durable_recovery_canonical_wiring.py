# © Artur Czarnecki. All rights reserved.

"""W3-C4 — canonical DECISION_DURABLE recovery entrypoint; no production bypass."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_CANONICAL_SYMBOL = "resume_decision_from_durable_state_with_recovery_admission"
_LEGACY_SYMBOL = "resume_decision_from_durable_state"

_ALLOWED_LEGACY_DEFINITION = frozenset(
    {
        "intergrax/runtime/execution/decision_recovery.py",
    },
)

_PRODUCTION_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "testing_support",
)


def _posix_relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _iter_production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            posix = path.as_posix()
            if "/tests/" in posix or path.name.startswith("test_"):
                continue
            if "/docker/runtime-context/" in posix:
                continue
            files.append(path)
    return files


def _legacy_symbol_line_hits(path: Path, relative: str) -> list[str]:
    hits: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if _LEGACY_SYMBOL not in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        hits.append(f"{relative}:{lineno}: {stripped[:160]}")
    return hits


def _legacy_symbol_hits(path: Path) -> list[str]:
    relative = _posix_relative(path)
    if relative in _ALLOWED_LEGACY_DEFINITION:
        return []
    text = path.read_text(encoding="utf-8-sig")
    if _LEGACY_SYMBOL not in text:
        return []
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError:
        return _legacy_symbol_line_hits(path, relative)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == _LEGACY_SYMBOL:
                    hits.append(f"{relative}:{node.lineno} import from {node.module}")
        if isinstance(node, ast.Name) and node.id == _LEGACY_SYMBOL:
            if isinstance(node.ctx, ast.Load):
                hits.append(f"{relative}:{node.lineno} reference")
        if isinstance(node, ast.Attribute) and node.attr == _LEGACY_SYMBOL:
            hits.append(f"{relative}:{node.lineno} attribute")
    return hits


def test_decision_durable_recovery_has_no_production_bypass() -> None:
    offenders: list[str] = []
    for path in _iter_production_python_files():
        offenders.extend(_legacy_symbol_hits(path))
    assert offenders == []


def test_canonical_handoff_symbol_is_defined() -> None:
    handoff = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "resilience"
        / "decision_durable_recovery_handoff.py"
    )
    text = handoff.read_text(encoding="utf-8")
    assert f"async def {_CANONICAL_SYMBOL}(" in text
