# © Artur Czarnecki. All rights reserved.

"""GR-10-R13 — ORCHESTRATION Reliability enterprise qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_RELIABILITY_INVENTORY,
    GR10_R13_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SLOT = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "orchestration"
    / "governed_consequential_operation.py"
)
_MSE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "orchestration_topology_slot_mse_enforcement.py"
)


def test_gr10_r13_orchestration_reliability_qualified() -> None:
    row = next(
        row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "Reliability"
    )
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("Reliability") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r13_reliability_inventory_has_no_gap_rows() -> None:
    assert GR10_ORCHESTRATION_RELIABILITY_INVENTORY
    for row in GR10_ORCHESTRATION_RELIABILITY_INVENTORY:
        assert row.coverage in {
            "QUALIFIED",
            "N/A",
            "delegated to another canonical owner",
            "QUALIFIED — delegated GR-7",
        }
        if row.production and row.consequential and row.coverage == "QUALIFIED":
            assert row.reliability_boundary
            assert row.idempotency


def test_gr10_r13_next_remediation_is_governance_evidence() -> None:
    assert GR10_R13_NEXT_REMEDIATION.capability == "Governance Evidence"
    assert "GR-10-R14" in GR10_R13_NEXT_REMEDIATION.task_name


def test_gr10_r13_governed_slot_wires_reliability_after_authorize() -> None:
    source = _SLOT.read_text(encoding="utf-8-sig")
    assert "effect_reliability" in source
    assert "execute_admitted_effect" in source
    assert "authorize_orchestration_consequential_effect" in source


def test_gr10_r13_production_topology_requires_reliability_policy() -> None:
    source = _MSE.read_text(encoding="utf-8-sig")
    assert "OrchestrationTopologyReliabilityCompositionError" in source
    tree = ast.parse(source, filename=str(_MSE))
    assert any(
        isinstance(node, ast.ClassDef)
        and node.name == "OrchestrationTopologyReliabilityCompositionError"
        for node in tree.body
    )
