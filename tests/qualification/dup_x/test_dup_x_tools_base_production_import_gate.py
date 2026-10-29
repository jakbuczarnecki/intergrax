# © Artur Czarnecki. All rights reserved.

"""DUP-X legacy tools_base architecture gate (Wave C)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)
_FORBIDDEN_MODULES = frozenset({"intergrax.tools.tools_base", "intergrax.tools.tools_base.ToolRegistry"})


def _production_py_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "/tests/" in rel or "\\tests\\" in rel or "/qualification/" in rel:
                continue
            if path.name == "tools_base.py":
                continue
            files.append(path)
    return files


def test_dup_x_no_new_production_imports_of_deprecated_tools_base() -> None:
    violators: list[str] = []
    for path in _production_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod == "intergrax.tools.tools_base" or mod.startswith("intergrax.tools.tools_base."):
                    violators.append(path.relative_to(_REPO_ROOT).as_posix())
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES or alias.name.startswith("intergrax.tools.tools_base"):
                        violators.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violators, sorted(set(violators))
