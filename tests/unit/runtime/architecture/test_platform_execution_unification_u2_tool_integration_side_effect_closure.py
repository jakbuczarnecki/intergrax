# © Artur Czarnecki. All rights reserved.

"""U2 — tool / integration side-effect closure static and contract gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPENSATION_WORKER = (
    _REPO_ROOT / "intergrax" / "agents" / "persistence" / "compensation_queue_worker.py"
)
_COMPENSATION_TOOL_SESSION = (
    _REPO_ROOT
    / "intergrax"
    / "agents"
    / "persistence"
    / "compensation_tool_invoke_session.py"
)
_EXECUTION_BOUND_INVOKER_CONTRACT = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "execution_bound_declarative_tool_invocation.py"
)
_COMPENSATION_SIDE_EFFECT_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "compensation_side_effect_wiring.py"
)
_CALLABLE_DECLARATIVE_INVOKER = (
    _REPO_ROOT
    / "intergrax"
    / "agents"
    / "persistence"
    / "declarative_tool_executor.py"
)
_U2_COMPENSATION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "compensation_side_effect_execution.py"
)
_U2_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md"
)


def test_u2_compensation_worker_uses_admitted_side_effect_port() -> None:
    source = _COMPENSATION_WORKER.read_text(encoding="utf-8")
    assert "CompensationSideEffectExecutionPort" in source
    assert "side_effect_execution" in source
    assert "DeclarativeToolInvoker" not in source
    tree = ast.parse(source)
    invoke_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(getattr(node.func, "attr", None), str)
        and node.func.attr == "invoke"
    ]
    assert invoke_calls == [], "compensation worker must not call tool invoker.invoke directly"


def test_u2_qualification_artifact_present() -> None:
    assert _U2_QUALIFICATION.is_file()


def test_u2_compensation_contract_has_no_any_on_public_boundary() -> None:
    source = _U2_COMPENSATION_CONTRACT.read_text(encoding="utf-8")
    assert "Any" not in source
    assert "dict[str, Any]" not in source


def test_u2_compensation_session_has_no_catalog_invoker_type_discrimination() -> None:
    source = _COMPENSATION_TOOL_SESSION.read_text(encoding="utf-8")
    assert "CatalogDeclarativeToolInvoker" not in source
    assert "CallableDeclarativeToolInvoker" not in source
    stripped = source.replace("ExecutionBoundDeclarativeToolInvoker", "")
    assert "DeclarativeToolInvoker" not in stripped
    assert "ExecutionBoundDeclarativeToolInvoker" in source
    for forbidden in ("getattr", "hasattr", "setattr"):
        assert forbidden not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "isinstance":
                raise AssertionError(
                    "compensation session must not use runtime isinstance discrimination",
                )


def test_u2_execution_bound_invoker_invoke_returns_typed_result() -> None:
    source = _EXECUTION_BOUND_INVOKER_CONTRACT.read_text(encoding="utf-8")
    assert "-> object" not in source.replace(" ", "")
    tree = ast.parse(source)
    invoke_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "invoke"
    ]
    assert invoke_methods, "ExecutionBoundDeclarativeToolInvoker must declare invoke"
    returns = invoke_methods[0].returns
    assert returns is not None
    if isinstance(returns, ast.Name):
        assert returns.id == "DeclarativeToolInvokeResult"
    elif isinstance(returns, ast.Constant) and isinstance(returns.value, str):
        assert returns.value == "DeclarativeToolInvokeResult"
    else:
        raise AssertionError("invoke must annotate DeclarativeToolInvokeResult return type")


def test_u2_compensation_wiring_requires_execution_bound_invoker() -> None:
    source = _COMPENSATION_SIDE_EFFECT_WIRING.read_text(encoding="utf-8")
    stripped = source.replace("ExecutionBoundDeclarativeToolInvoker", "")
    assert "DeclarativeToolInvoker" not in stripped
    assert "ExecutionBoundDeclarativeToolInvoker" in source


def _class_method_names(tree: ast.Module, class_name: str) -> set[str]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def test_u2_callable_declarative_invoker_must_not_fake_execution_binding() -> None:
    source = _CALLABLE_DECLARATIVE_INVOKER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = _class_method_names(tree, "CallableDeclarativeToolInvoker")
    assert "bind_execution_identity" not in methods, (
        "CallableDeclarativeToolInvoker must not implement fake execution-bound binding"
    )
