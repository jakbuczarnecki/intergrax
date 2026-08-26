# © Artur Czarnecki. All rights reserved.

"""Architecture gate — runtime diagnostics must not import hosting (HOST-DIAG-2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCAN_ROOT = "intergrax/runtime/diagnostics"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.hosting",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _diagnostics_python_files() -> list[Path]:
    root = _repo_root()
    base = root / _SCAN_ROOT
    return sorted(base.rglob("*.py"))


def _collect_forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if node.module == prefix or node.module.startswith(f"{prefix}."):
                    violations.append(f"{rel}:{node.lineno} imports from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                    if alias.name == prefix or alias.name.startswith(f"{prefix}."):
                        violations.append(f"{rel}:{node.lineno} imports {alias.name}")
    return violations


def test_runtime_diagnostics_does_not_import_hosting() -> None:
    violations: list[str] = []
    for path in _diagnostics_python_files():
        violations.extend(_collect_forbidden_imports(path))
    assert violations == []
