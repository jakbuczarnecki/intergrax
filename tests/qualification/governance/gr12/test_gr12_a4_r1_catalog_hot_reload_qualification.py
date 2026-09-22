# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R1 — catalog hot reload CLA-04 qualification gates (superseded by R1-R1 SSOT)."""

from __future__ import annotations

import pytest

from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R1_R1_QUALIFICATION_PROOF,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr12_a4_r1_catalog_hot_reload_qualified_ssot() -> None:
    row = next(
        item for item in GR12_CONTROL_PLANE_SURFACES if item.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert row.coverage is Gr12CoverageStatus.QUALIFIED
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert GR12_A4_R1_R1_QUALIFICATION_PROOF in row.qualification_proof
