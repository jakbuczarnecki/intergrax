# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R2 — architecture proof for mandatory progress boundary wiring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"
_BOUNDARY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"


def _function_body_calls(source_path: Path, function_name: str, callee: str) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == callee:
                    return True
    return False


def test_execution_runtime_wires_continuation_store_into_boundary() -> None:
    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert "continuation_state_store=self._continuation_state_store" in runtime_source
    assert "bind_active_execution_continuation_state_store" in runtime_source


def test_execution_boundary_run_path_enforces_progress_gate() -> None:
    assert _function_body_calls(
        _BOUNDARY,
        "_run_admission_and_delegate",
        "assert_canonical_execution_may_progress",
    )
