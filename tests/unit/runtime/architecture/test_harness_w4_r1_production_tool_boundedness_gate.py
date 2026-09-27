# © Artur Czarnecki. All rights reserved.

"""HARNESS-W4-R1 — production tool boundedness architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.nexus.tools.runtime_tool_invoker_composition import (
    ProductionRuntimeToolInvokerCompositionError,
    build_production_runtime_tool_invoker,
)
from tests.unit.runtime.nexus.tools.conftest import FakeRegistry
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "declarative_tool_wiring.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "execution_bound_catalog_tool_composition.py",
)

_COMPOSITION_BUILDER = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "runtime_tool_invoker_composition.py"
)


class _In(BaseModel):
    value: int


class _Out(BaseModel):
    value: int


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _production_builder_calls_missing_boundary(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "build_production_runtime_tool_invoker":
            continue
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "dependency_attempt_boundary" not in keywords:
            offenders.append(f"{rel}:{node.lineno}")
    return offenders


def test_w4_r1_production_roots_pass_dependency_attempt_boundary() -> None:
    offenders: list[str] = []
    for path in _PRODUCTION_ROOTS:
        offenders.extend(_production_builder_calls_missing_boundary(path))
    assert offenders == [], (
        "production roots must pass dependency_attempt_boundary to "
        "build_production_runtime_tool_invoker:\n" + "\n".join(offenders)
    )


def test_w4_r1_central_builder_enforces_production_boundary_ast() -> None:
    tree = ast.parse(
        _COMPOSITION_BUILDER.read_text(encoding="utf-8"),
        filename=str(_COMPOSITION_BUILDER),
    )
    source = ast.unparse(tree)
    assert "production_mode" in source and "dependency_attempt_boundary is None" in source


def test_w4_r1_central_builder_enforces_production_boundary_behavior() -> None:
    registry = FakeRegistry(
        ToolContract(
            tool_id="gate.tool",
            name="gate",
            description="gate",
            input_schema=_In,
            output_schema=_Out,
            side_effects=False,
            error_mapping={},
            risk_level=ToolRiskLevel.LOW,
        ),
    )
    with pytest.raises(ProductionRuntimeToolInvokerCompositionError):
        build_production_runtime_tool_invoker(registry=registry, production_mode=True)
