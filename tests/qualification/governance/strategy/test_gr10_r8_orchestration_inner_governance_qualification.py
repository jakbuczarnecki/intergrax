# © Artur Czarnecki. All rights reserved.

"""GR-10-R8 — ORCHESTRATION Inner Governance enterprise qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.runtime.architecture.gr3_inner_enforcement_ast import (
    collect_forbidden_concrete_inner_guard_imports,
    prepare_invocation_inner_guard_before_authorization_indices,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_GEP_COVERAGE_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_R8_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_CONTEXT = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py"
)
_DECLARATIVE_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "declarative_tool_wiring.py"
)
_COMPOSITION = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "runtime_tool_invoker_composition.py"
)
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"


def test_gr10_r8_orchestration_inner_governance_qualified() -> None:
    inner = next(
        row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "Inner Governance"
    )
    assert inner.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("Inner Governance") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r8_tool_invocation_authorization_gep_qualified() -> None:
    row = next(row for row in GR10_GEP_COVERAGE_INVENTORY if row.gep == "TOOL_INVOCATION_AUTHORIZATION")
    assert row.orchestration_coverage is Gr10CoverageStatus.QUALIFIED


def test_gr10_r8_production_wiring_uses_composition_builder() -> None:
    ctx_source = _RUNTIME_CONTEXT.read_text(encoding="utf-8-sig")
    assert "build_production_runtime_tool_invoker(" in ctx_source
    assert "inner_execution_guard=config.canonical_inner_execution_guard" in ctx_source
    decl_source = _DECLARATIVE_WIRING.read_text(encoding="utf-8-sig")
    assert "build_production_runtime_tool_invoker(" in decl_source
    assert "inner_execution_guard=canonical_inner_execution_guard" in decl_source
    for path in (_RUNTIME_CONTEXT, _DECLARATIVE_WIRING):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = path.relative_to(_REPO_ROOT).as_posix()
        assert collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel) == []


def test_gr10_r8_composition_requires_guard_on_production_mode_ast() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_default_canonical_inner_execution_guard" in source
    assert "production_mode" in source
    tree = ast.parse(source, filename=str(_COMPOSITION))
    assert "RuntimeToolInvoker(" in source


def test_gr10_r8_invoker_calls_inner_guard_before_authorization_ast() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_INVOKER))
    guard_idx, auth_idx = prepare_invocation_inner_guard_before_authorization_indices(tree)
    assert guard_idx is not None and auth_idx is not None
    assert guard_idx < auth_idx
    assert "guard.assert_meaningful_side_effect_bound" in source


def test_gr10_r8_next_remediation_is_mse_not_inner() -> None:
    assert GR10_R8_NEXT_REMEDIATION.capability == "MSE"
    assert GR10_R8_NEXT_REMEDIATION.strategy == "ORCHESTRATION"
    assert "GR-10-R9" in GR10_R8_NEXT_REMEDIATION.task_name
