# © Artur Czarnecki. All rights reserved.

"""UE-10R3 — platform-owned root identity public facade gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.execution import __all__ as execution_public_api

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FACADE_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "facade.py"


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _function_node(path: Path, class_name: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                return item
    raise AssertionError(f"{class_name}.{name} not found in {path}")


def _execute_arg_names(execute_node: ast.AsyncFunctionDef) -> list[str]:
    names: list[str] = []
    args = execute_node.args
    for arg in args.posonlyargs + args.args:
        names.append(arg.arg)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    for arg in args.kwonlyargs:
        names.append(arg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def test_facade_execute_does_not_accept_root_execution_context() -> None:
    execute_node = _function_node(_FACADE_PATH, "Execution", "execute")
    arg_names = _execute_arg_names(execute_node)
    assert "root_context" not in arg_names
    assert "options" in arg_names


def test_facade_execute_resolves_root_execution_context() -> None:
    execute_node = _function_node(_FACADE_PATH, "Execution", "execute")
    resolved = False
    for node in ast.walk(execute_node):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == "resolve_root_execution_context":
            resolved = True
            break
    assert resolved, "Execution.execute must call resolve_root_execution_context()"


def test_package_exports_root_execution_options_not_context() -> None:
    assert "RootExecutionOptions" in execution_public_api
    assert "RootExecutionContext" not in execution_public_api
