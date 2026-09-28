# © Artur Czarnecki. All rights reserved.

"""GR-13 mechanical proof matrix gates (G13-01..G13-14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gr13.catalog import (
    GR13_BASELINE_HEAD,
    GR13_PROOF_MATRIX,
    Gr13MatrixResult,
    gr13_all_proof_nodes,
)
from tests.qualification.governance.strategy.catalog import (
    GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED,
    GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED,
    Gr10EvidenceCertificationRequirement,
    gr10_agentic_gep_semantics,
    gr10_orchestration_gep_semantics,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_g13_01_closed_world_inventory_matches_gr10_deferred() -> None:
    agentic = {row.gep for row in GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED}
    orch = {row.gep for row in GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED}
    matrix_agentic = {row.gep for row in GR13_PROOF_MATRIX if row.strategy == "AGENTIC"}
    matrix_orch = {
        row.gep for row in GR13_PROOF_MATRIX if row.strategy == "ORCHESTRATION"
    }
    assert matrix_agentic == agentic
    assert matrix_orch == orch
    assert len(agentic) == 7
    assert len(orch) == 5


def test_g13_02_all_required_rows_qualified() -> None:
    for row in GR13_PROOF_MATRIX:
        sem = (
            gr10_agentic_gep_semantics(row.gep)
            if row.strategy == "AGENTIC"
            else gr10_orchestration_gep_semantics(row.gep)
        )
        assert (
            sem.gr13_evidence_requirement
            is Gr10EvidenceCertificationRequirement.REQUIRED_IN_GR13
        )
        assert row.result is Gr13MatrixResult.QUALIFIED


def test_g13_03_canonical_fact_contract_on_all_rows() -> None:
    for row in GR13_PROOF_MATRIX:
        assert "GovernanceDecisionEvidenceFact" in row.fact_contract


def test_g13_04_canonical_persistence_contract_on_all_rows() -> None:
    for row in GR13_PROOF_MATRIX:
        assert "GovernanceEvidencePersistencePort" in row.persistence_contract


_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_g13_12_proof_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        gr13_all_proof_nodes(),
        _REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def test_g13_baseline_head_documented() -> None:
    assert GR13_BASELINE_HEAD.startswith("4e5878f")
