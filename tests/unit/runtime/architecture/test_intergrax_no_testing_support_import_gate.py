# © Artur Czarnecki. All rights reserved.

"""Architecture gate — ``intergrax/`` must not import ``testing_support``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_DIR = _REPO_ROOT / "intergrax"

_FORBIDDEN_ROOT = "testing_support"


def _import_root_module(module: str | None) -> str | None:
    if module is None:
        return None
    return module.split(".", 1)[0]


def _is_forbidden_import(module: str) -> bool:
    root = _import_root_module(module)
    return root == _FORBIDDEN_ROOT or module.startswith("testing_support.")


def _collect_forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_REPO_ROOT).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_import(alias.name):
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if _is_forbidden_import(node.module):
                violations.append(f"{rel}:{node.lineno} imports from {node.module}")
    return violations


def _intergrax_testing_support_import_violations() -> list[str]:
    violations: list[str] = []
    for path in sorted(_INTERGRAX_DIR.rglob("*.py")):
        violations.extend(_collect_forbidden_imports(path))
    return violations


def test_intergrax_package_does_not_import_testing_support() -> None:
    violations = _intergrax_testing_support_import_violations()
    assert violations == [], (
        "intergrax/ must not import testing_support; violations: "
        + "; ".join(violations)
    )


def test_intergrax_testing_support_gate_detects_synthetic_violation() -> None:
    sample = "from testing_support.foo import Bar\n"
    tree = ast.parse(sample, filename="<sample>")
    node = tree.body[0]
    assert isinstance(node, ast.ImportFrom)
    assert node.module is not None
    assert _is_forbidden_import(node.module)
