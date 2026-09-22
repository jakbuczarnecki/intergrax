# © Artur Czarnecki. All rights reserved.

"""CHR-13 — production catalog hot-reload bypass inventory."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GOVERNED_SERVICE = "intergrax/applications/_shared/catalog_hot_reload_service.py"
_BLOCKED_RELOAD = "intergrax/integrations/registry/catalog_hot_reload.py"
_ALLOWED_CALLERS = frozenset({_GOVERNED_SERVICE, _BLOCKED_RELOAD})


def _calls_reload_integration_catalog(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "reload_integration_catalog":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "reload_integration_catalog":
            return True
    return False


def _production_reload_bypass_offenders() -> list[str]:
    roots = (_REPO_ROOT / "intergrax", _REPO_ROOT / "applications")
    offenders: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            posix = path.as_posix()
            if "/tests/" in posix or path.name.startswith("test_"):
                continue
            if "/docker/runtime-context/" in posix:
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            if relative in _ALLOWED_CALLERS:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            if _calls_reload_integration_catalog(tree):
                offenders.append(relative)
    return sorted(offenders)


def test_chr13_no_production_bypass_reload_callers() -> None:
    assert _production_reload_bypass_offenders() == []
