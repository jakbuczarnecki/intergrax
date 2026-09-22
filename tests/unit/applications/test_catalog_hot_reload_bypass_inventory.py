# © Artur Czarnecki. All rights reserved.

"""CHR-13 — production catalog hot-reload bypass inventory."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_WIRING = "intergrax/applications/_shared/catalog_hot_reload_wiring.py"
_CATALOG_SERVICE = "intergrax/applications/_shared/catalog_hot_reload_service.py"
_BLOCKED_RELOAD = "intergrax/integrations/registry/catalog_hot_reload.py"
_ALLOWED_CALLERS = frozenset({_CATALOG_SERVICE, _BLOCKED_RELOAD})
_BOOTSTRAP_REGISTER = frozenset(
    {
        "intergrax/integrations/registry/bootstrap_core.py",
        "intergrax/integrations/registry/bootstrap_extended.py",
        "intergrax/integrations/registry/plugin_register.py",
    }
)


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


def _production_register_integration_callers() -> set[str]:
    callers: set[str] = set()
    roots = (_REPO_ROOT / "intergrax", _REPO_ROOT / "applications")
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
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == "register_integration":
                    callers.add(relative)
                if isinstance(func, ast.Attribute) and func.attr == "register_integration":
                    callers.add(relative)
    return callers


def test_chr13_no_production_bypass_reload_callers() -> None:
    assert _production_reload_bypass_offenders() == []


def test_chr14_catalog_hot_reload_wiring_does_not_construct_request_identity() -> None:
    source = (_REPO_ROOT / _CATALOG_WIRING).read_text(encoding="utf-8")
    assert "RequestIdentity(" not in source
    assert "RequestIdentity" not in source


def test_chr15_catalog_hot_reload_service_does_not_construct_request_identity() -> None:
    source = (_REPO_ROOT / _CATALOG_SERVICE).read_text(encoding="utf-8")
    assert "RequestIdentity(" not in source


def test_chr16_live_operator_catalog_mutation_paths_only_governed_reload() -> None:
    callers = _production_register_integration_callers()
    live_operator = callers - _BOOTSTRAP_REGISTER - {
        "intergrax/integrations/registry/catalog.py",
        "intergrax/applications/_shared/catalog_hot_reload_service.py",
    }
    assert live_operator == set()
