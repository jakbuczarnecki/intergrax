# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R3 — continuation identity must not use Task runtime authority."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOUNDARY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"
_CHILD = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"


def _forbidden_task_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.task"):
                modules.append(node.module)
    return modules


def test_execution_boundary_has_no_task_layer_import() -> None:
    assert _forbidden_task_imports(_BOUNDARY) == []


def test_child_execution_runner_has_no_task_layer_import() -> None:
    assert _forbidden_task_imports(_CHILD) == []


def test_boundary_requires_execution_continuation_identity_helper() -> None:
    source = _BOUNDARY.read_text(encoding="utf-8")
    assert "require_execution_continuation_identity" in source
    assert "ActiveTaskRegistry" not in source
