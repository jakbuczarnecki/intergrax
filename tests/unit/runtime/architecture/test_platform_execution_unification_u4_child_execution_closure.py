# © Artur Czarnecki. All rights reserved.

"""U4 — delegated subtask child execution closure static and contract gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FACTORY_RUNTIME = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "production_agent_capability_runtime.py"
)
_CHILD_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "production_delegated_subtask_child_execution_wiring.py"
)
_AC4_E2E = (
    _REPO_ROOT / "tests" / "unit" / "applications" / "test_ac4_phase9_production_composition_e2e.py"
)
_U4_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U4_CHILD_EXECUTION_CLOSURE.md"
)
_WORK_PORT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "execution_work_port.py"


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _factory_create_child_execution_optional(factory_source: str) -> bool:
    tree = ast.parse(factory_source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "DelegatedSubtaskServiceFactory":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "create":
                continue
            for arg in item.args.kwonlyargs:
                if arg.arg == "child_execution" and arg.annotation is None:
                    return True
                if arg.arg != "child_execution":
                    continue
                ann = arg.annotation
                if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
                    if isinstance(ann.right, ast.Constant) and ann.right.value is None:
                        return True
    return False


def _constructs_child_execution_runner(source: str, *, filename: str) -> bool:
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "ChildExecutionRunner":
            return True
        if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name):
            if func.value.id == "ChildExecutionRunner":
                return True
    return False


def test_u4_qualification_artifact_present() -> None:
    assert _U4_QUALIFICATION.is_file()


def test_u4_factory_requires_child_execution_and_does_not_construct_runner() -> None:
    source = _FACTORY_RUNTIME.read_text(encoding="utf-8")
    assert "ChildExecutionRunner" not in source
    assert "as_child_execution_port" not in source
    assert not _factory_create_child_execution_optional(source)
    assert not _constructs_child_execution_runner(source, filename=_rel(_FACTORY_RUNTIME))


def test_u4_composition_root_wires_canonical_child_execution_adapter() -> None:
    wiring = _CHILD_WIRING.read_text(encoding="utf-8")
    assert "build_production_delegated_subtask_child_execution_port" in wiring
    assert "as_child_execution_port" in wiring
    assert _constructs_child_execution_runner(wiring, filename=_rel(_CHILD_WIRING))
    work_port = _WORK_PORT.read_text(encoding="utf-8")
    assert "class ChildExecutionWorkPort" in work_port
    assert "ChildExecutionRunner" in work_port


def test_u4_production_e2e_passes_composition_root_child_port() -> None:
    source = _AC4_E2E.read_text(encoding="utf-8")
    assert "delegated_subtask_child_execution.port()" in source
    assert "child_execution=capability_runtime.delegated_subtask_child_execution.port()" in source


def test_u4_capability_runtime_exposes_delegated_subtask_child_execution_binding() -> None:
    source = _FACTORY_RUNTIME.read_text(encoding="utf-8")
    assert "delegated_subtask_child_execution: ProductionDelegatedSubtaskChildExecutionPort" in source
    assert "build_production_delegated_subtask_child_execution_port" in source
