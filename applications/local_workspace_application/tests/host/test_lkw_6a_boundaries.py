# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_LKW_ROOT = Path(__file__).resolve().parents[2]

# Product-owned LKW.7 sidecar is allowed. Platform-generic OS service install
# and vendor Slack Socket Mode transport are not owned by LKW application code.
_FORBIDDEN_IMPORT_ROOTS = (
    "slack_sdk",
    "win32service",
    "win32serviceutil",
    "servicemanager",
)
_FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"\bSocketModeClient\b"),
    re.compile(r"\bSocketModeHandler\b"),
    re.compile(r"\bWin32Service\b"),
    re.compile(r"\bwin32serviceutil\b"),
)


def _iter_lkw_implementation_py() -> list[Path]:
    paths: list[Path] = []
    for path in _LKW_ROOT.rglob("*.py"):
        if "tests" in path.parts or "docs" in path.parts:
            continue
        paths.append(path)
    return paths


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_no_lkw_specific_interaction_adapter_hierarchy() -> None:
    violations: list[str] = []
    for path in _iter_lkw_implementation_py():
        text = path.read_text(encoding="utf-8")
        if "class Lkw" in text and "Interaction" in text:
            violations.append(str(path))
        if "class LocalWorkspace" in text and "InteractionAdapter" in text:
            violations.append(str(path))
    assert violations == []


def test_no_os_service_or_slack_socket_or_file_watcher_in_lkw_implementation() -> None:
    """LKW may compose/configure platform capabilities; it must not own vendor
    Slack Socket Mode transport or OS service-install APIs.
    Application-owned LKW.7 ``file_watcher`` composition remains allowed.
    """
    violations: list[str] = []
    for path in _iter_lkw_implementation_py():
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: unparseable ({exc})")
            continue
        imported = _imported_roots(tree)
        for root in _FORBIDDEN_IMPORT_ROOTS:
            if root in imported:
                violations.append(f"{path}: forbidden import root {root}")
        for pattern in _FORBIDDEN_NAME_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path}: forbidden symbol {pattern.pattern}")
    assert violations == []


def test_lkw_does_not_import_vendor_slack_sdk() -> None:
    for path in _iter_lkw_implementation_py():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = _imported_roots(tree)
        assert "slack_sdk" not in imported, path
