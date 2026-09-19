# © Artur Czarnecki. All rights reserved.

"""GR-10-R9-R2 — orchestration graph/non-tool MSE architecture gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
GRAPH_EXECUTOR = REPO_ROOT / "intergrax/runtime/nexus/execution/graph_executor.py"
GOVERNED_OP = REPO_ROOT / "intergrax/runtime/nexus/orchestration/governed_consequential_operation.py"
GRAPH_MSE = REPO_ROOT / "intergrax/runtime/nexus/orchestration/orchestration_graph_meaningful_side_effect.py"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_gr10_r9_r2_inventory_has_no_b_classification() -> None:
    for row in GR10_ORCHESTRATION_MSE_NON_TOOL_INVENTORY:
        assert "B —" not in row.classification
        assert row.classification.strip()
        assert row.classification != "TBD"


def test_gr10_r9_r2_orchestration_mse_matrix_qualified() -> None:
    row = next(r for r in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if r.capability == "MSE")
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("MSE") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r9_r2_graph_executor_no_direct_provider_mutation() -> None:
    source = _read(GRAPH_EXECUTOR)
    forbidden = (
        "provider.",
        "ExternalWorkAdapter",
        "authorize_and_execute(",
    )
    for token in forbidden:
        assert token not in source


def test_gr10_r9_r2_governed_operation_depends_on_port_not_boundary() -> None:
    source = _read(GOVERNED_OP)
    assert "MeaningfulSideEffectAuthorizationPort" in source
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source
    assert "if mse_port is None" not in source.replace(" ", "")
    assert re.search(r"if\s+meaningful_side_effect_authorization\s+is\s+None\s*:\s*\n\s*return", source) is None


def test_gr10_r9_r2_graph_projection_uses_canonical_identity() -> None:
    source = _read(GRAPH_MSE)
    assert "require_active_execution_governance_identity" in source
    assert "NestedOrchestrationGovernanceIdentity" not in source
    tree = ast.parse(source, filename=str(GRAPH_MSE))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "MeaningfulSideEffectAuthorizationBoundary" not in (node.module or "")
