# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R1-R1 — catalog revision authority and operator identity qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_NEXT_REMEDIATION,
    GR12_A4_R1_R1_EXECUTION_PROOF_NODES,
    GR12_A4_R1_R1_QUALIFICATION_PROOF,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.qualification_support import assert_proof_nodes_registered

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a4_r1_r1_catalog_hot_reload_qualified_ssot() -> None:
    row = next(
        item for item in GR12_CONTROL_PLANE_SURFACES if item.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert row.coverage is Gr12CoverageStatus.QUALIFIED
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert row.future_remediation == ""
    assert GR12_A4_R1_R1_QUALIFICATION_PROOF in row.qualification_proof
    assert "GR-12-A4-R2-R1" in GR12_A4_NEXT_REMEDIATION.task_name


def test_gr12_a4_r1_r1_execution_proof_nodes_bound_to_semantic_tests() -> None:
    assert_proof_nodes_registered(GR12_A4_R1_R1_EXECUTION_PROOF_NODES)
