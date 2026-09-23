# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2-R1 — governed vector operator service qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_NEXT_REMEDIATION,
    GR12_A4_R2_R1_EXECUTION_PROOF_NODES,
    GR12_A4_R2_R1_QUALIFICATION_PROOF,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.qualification_support import assert_proof_nodes_registered

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a4_r2_r1_vector_operator_qualified_ssot() -> None:
    row = next(
        item for item in GR12_CONTROL_PLANE_SURFACES if item.path_id == "CP-VECTOR-INDEX-ADMIN"
    )
    assert row.coverage is Gr12CoverageStatus.QUALIFIED
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert row.future_remediation == ""
    assert GR12_A4_R2_R1_QUALIFICATION_PROOF in row.qualification_proof
    assert "VectorIndexAdminService" in row.production_entrypoint
    assert "GR-12-A4-R3" in GR12_A4_NEXT_REMEDIATION.task_name


def test_gr12_a4_r2_r1_execution_proof_nodes_bound_to_semantic_tests() -> None:
    assert_proof_nodes_registered(GR12_A4_R2_R1_EXECUTION_PROOF_NODES)


def test_gr12_a4_r2_r1_tenant_authority_proof_nodes_present() -> None:
    tenant_nodes = (
        node
        for node in GR12_A4_R2_R1_EXECUTION_PROOF_NODES
        if "tenant_mismatch" in node or "same_tenant" in node
    )
    assert any(tenant_nodes)


def test_gr12_a4_r2_r1_authority_to_execution_proof_node_present() -> None:
    binding_nodes = (
        node
        for node in GR12_A4_R2_R1_EXECUTION_PROOF_NODES
        if "authority_to_execution" in node
    )
    assert any(binding_nodes)
