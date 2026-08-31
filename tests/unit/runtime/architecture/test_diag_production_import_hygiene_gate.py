# © Artur Czarnecki. All rights reserved.

"""Architecture gate — production diagnostics must not import dev/test-only modules."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DIAGNOSTICS_DIR = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"

_FORBIDDEN_ROOT_MODULES = frozenset(
    {
        "pytest",
        "hypothesis",
        "unittest",
        "tests",
        "testing_support",
    }
)


def _import_root_module(module: str | None) -> str | None:
    if module is None:
        return None
    return module.split(".", 1)[0]


def _is_forbidden_import(module: str) -> bool:
    root = _import_root_module(module)
    if root is None:
        return False
    if root in _FORBIDDEN_ROOT_MODULES:
        return True
    return module == "unittest.mock" or module.startswith("unittest.")


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


def _diagnostics_forbidden_import_violations() -> list[str]:
    violations: list[str] = []
    for path in sorted(_DIAGNOSTICS_DIR.rglob("*.py")):
        violations.extend(_collect_forbidden_imports(path))
    return violations


def test_diagnostics_production_modules_do_not_import_dev_dependencies() -> None:
    violations = _diagnostics_forbidden_import_violations()
    assert violations == [], (
        "runtime/diagnostics production modules must not import dev/test-only dependencies: "
        + "; ".join(violations)
    )


def test_diagnostics_package_imports_without_pytest_subprocess() -> None:
    script = """
import builtins
import sys

_real_import = builtins.__import__

def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pytest" or name.startswith("pytest."):
        raise ModuleNotFoundError(f"No module named {name!r}")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import
sys.modules.pop("pytest", None)
import intergrax.runtime.diagnostics  # noqa: F401
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
