# © Artur Czarnecki. All rights reserved.

"""GR-10-R9 / ADR1 — ORCHESTRATION MSE qualification honesty (tool slice vs strategy row)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_GEP_COVERAGE_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_R9_NEXT_REMEDIATION,
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


def test_gr10_r9_orchestration_mse_partial_until_r9_r1() -> None:
    row = next(row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.PARTIAL
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.PARTIAL


def test_gr10_r9_meaningful_side_effect_gep_orchestration_partial() -> None:
    row = next(row for row in GR10_GEP_COVERAGE_INVENTORY if row.gep == "MEANINGFUL_SIDE_EFFECT")
    assert row.orchestration_coverage is Gr10CoverageStatus.PARTIAL


def test_gr10_r9_production_wiring_uses_composition_builder() -> None:
    ctx_source = _RUNTIME_CONTEXT.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization=config.meaningful_side_effect_authorization" in ctx_source
    decl_source = _DECLARATIVE_WIRING.read_text(encoding="utf-8-sig")
    assert "meaningful_side_effect_authorization=" in decl_source
    comp_source = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_default_orchestration_meaningful_side_effect_authorization_boundary" in comp_source


def test_gr10_r9_invoker_enforces_boundary_before_effect_ast() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    assert "_require_canonical_meaningful_side_effect_authorization" in source
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source
    tree = ast.parse(source, filename=str(_INVOKER))
    config_source = _CONFIG.read_text(encoding="utf-8-sig")
    assert "MeaningfulSideEffectAuthorizationPort" in config_source
    assert "meaningful_side_effect_authorization" in config_source
    assert tree is not None


def test_gr10_r9_next_remediation_is_r9_r1_mse() -> None:
    assert GR10_R9_NEXT_REMEDIATION.capability == "MSE"
    assert "GR-10-R9-R1" in GR10_R9_NEXT_REMEDIATION.task_name
