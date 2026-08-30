# © Artur Czarnecki. All rights reserved.

"""UE-10R2 — single canonical root ExecutionId architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_node(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ExecutionRuntime":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return item
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _collect_mint_execution_id_calls_in_node(node: ast.AST) -> list[int]:
    lines: list[int] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _call_name(child.func) == "mint_execution_id":
            lines.append(child.lineno)
    return lines


def test_execution_runtime_execute_does_not_mint_execution_id() -> None:
    execute_node = _function_node(_RUNTIME_PATH, "execute")
    violations = _collect_mint_execution_id_calls_in_node(execute_node)
    assert violations == [], (
        "ExecutionRuntime.execute must not mint ExecutionId: "
        + ", ".join(f"runtime.py:{line}" for line in violations)
    )


def test_runtime_module_mints_execution_id_only_in_mint_root_execution_identity() -> None:
    mint_root_node = _function_node(_RUNTIME_PATH, "mint_root_execution_identity")
    allowed_lines = set(_collect_mint_execution_id_calls_in_node(mint_root_node))

    tree = ast.parse(_RUNTIME_PATH.read_text(encoding="utf-8-sig"), filename=str(_RUNTIME_PATH))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "mint_execution_id":
            continue
        if node.lineno not in allowed_lines:
            rel = _RUNTIME_PATH.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}:{node.lineno}: mint_execution_id()")
    assert violations == [], (
        "runtime.py must mint ExecutionId only in mint_root_execution_identity: "
        + ", ".join(violations)
    )
