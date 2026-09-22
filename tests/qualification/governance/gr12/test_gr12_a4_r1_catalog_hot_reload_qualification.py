# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R1 — catalog hot reload CLA-04 qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_NEXT_REMEDIATION,
    GR12_A4_R1_QUALIFICATION_PROOF,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a4_r1_catalog_hot_reload_wired_not_qualified_ssot() -> None:
    row = next(
        item for item in GR12_CONTROL_PLANE_SURFACES if item.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert GR12_A4_R1_QUALIFICATION_PROOF in row.qualification_proof
    assert "GR-12-A4-R1-R1" in row.future_remediation
    assert "GR-12-A4-R1-R1" in GR12_A4_NEXT_REMEDIATION.task_name
    blocker = GR12_A4_NEXT_REMEDIATION.exact_blocker.lower()
    assert "revision" in blocker and "authoritative" in blocker
    assert "aba" in blocker
    assert "requestidentity" in blocker.replace(" ", "") or "request identity" in blocker


def test_gr12_a4_r1_governance_proofs_importable() -> None:
    from tests.unit.applications import test_catalog_hot_reload_governance
    from tests.unit.applications import test_catalog_hot_reload_bypass_inventory

    assert test_catalog_hot_reload_governance is not None
    assert test_catalog_hot_reload_bypass_inventory is not None
