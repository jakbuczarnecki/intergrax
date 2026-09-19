# © Artur Czarnecki. All rights reserved.

"""GR-10-R9 / ADR1 / R9-R1 — ORCHESTRATION MSE qualification honesty."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_GEP_COVERAGE_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY,
    GR10_R9_R1_NEXT_REMEDIATION,
    GR10_R9_R2_NEXT_REMEDIATION,
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
_CONFIG = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "config.py"
_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "meaningful_side_effect_authorization.py"


def test_gr10_r9_orchestration_mse_qualified_after_r9_r2_graph_closure() -> None:
    row = next(row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r9_meaningful_side_effect_gep_orchestration_qualified() -> None:
    row = next(row for row in GR10_GEP_COVERAGE_INVENTORY if row.gep == "MEANINGFUL_SIDE_EFFECT")
    assert row.orchestration_coverage is Gr10CoverageStatus.QUALIFIED


def test_gr10_r9_production_wiring_requires_explicit_mse_port() -> None:
    ctx_source = _RUNTIME_CONTEXT.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization=config.meaningful_side_effect_authorization" in ctx_source
    decl_source = _DECLARATIVE_WIRING.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization=" in decl_source
    comp_source = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization is required when production_mode=True" in comp_source
    assert "build_default_orchestration_meaningful_side_effect_authorization_boundary" not in comp_source


def test_gr10_r9_invoker_enforces_boundary_before_effect_ast() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    assert "_require_canonical_meaningful_side_effect_authorization" in source
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source
    tree = ast.parse(source, filename=str(_INVOKER))
    config_source = _CONFIG.read_text(encoding="utf-8-sig")
    assert "MeaningfulSideEffectAuthorizationPort" in config_source
    assert "meaningful_side_effect_authorization" in config_source
    contract_source = _CONTRACT.read_text(encoding="utf-8-sig")
    assert "class MeaningfulSideEffectAuthorizationPort" in contract_source
    assert tree is not None


def test_gr10_r9_r1_next_remediation_is_graph_mse_follow_up() -> None:
    assert GR10_R9_R1_NEXT_REMEDIATION.capability == "MSE"
    assert "GR-10-R9" in GR10_R9_R1_NEXT_REMEDIATION.task_name


def test_gr10_r9_r2_inventory_zero_b_gap() -> None:
    assert GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY
    for row in GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY:
        assert "B —" not in row.classification


def test_gr10_r9_r2_next_remediation_is_decision_bound() -> None:
    assert GR10_R9_R2_NEXT_REMEDIATION.capability == "Decision-bound effect"
    assert "GR-10-R10" in GR10_R9_R2_NEXT_REMEDIATION.task_name
