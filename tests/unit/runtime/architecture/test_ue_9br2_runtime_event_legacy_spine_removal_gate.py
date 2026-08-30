# © Artur Czarnecki. All rights reserved.

"""UE-9BR2 — production event runtime must not retain legacy spine compatibility."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVENT_RUNTIME_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "events"
_OBS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "observability"

_LEGACY_SYMBOLS = frozenset(
    {
        "LEGACY_SPINE_TO_PLATFORM_KIND",
        "migrate_legacy_spine_payload",
        "legacy_spine_type",
        "legacy_spine_value",
    }
)

_EXCLUDED_PARTS = frozenset({"__pycache__", "tests"})
_CONFORMANCE_EXCLUDED = frozenset({"persistence_conformance.py"})


def _scan_roots() -> list[Path]:
    roots: list[Path] = [_EVENT_RUNTIME_ROOT, _OBS_ROOT]
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if path.name in _CONFORMANCE_EXCLUDED:
                continue
            files.append(path)
    return files


def _collect_legacy_symbol_hits() -> list[str]:
    violations: list[str] = []
    for path in _scan_roots():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for symbol in _LEGACY_SYMBOLS:
                if symbol in line:
                    violations.append(f"{rel}:{lineno}: {symbol}")
    return violations


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def test_production_event_runtime_has_zero_legacy_spine_symbols() -> None:
    assert _collect_legacy_symbol_hits() == []


def test_parse_runtime_event_payload_is_canonical_only() -> None:
    path = _EVENT_RUNTIME_ROOT / "runtime_event.py"
    assert _collect_forbidden_calls(path, frozenset({"migrate_legacy_spine_payload"})) == []
