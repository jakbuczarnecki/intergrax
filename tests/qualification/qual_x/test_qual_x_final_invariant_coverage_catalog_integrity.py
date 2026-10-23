# © Artur Czarnecki. All rights reserved.

"""QUAL-X-FINAL — closed-world catalog provenance and REG-01 credibility gates."""

from __future__ import annotations

import pytest

from tests.qualification.qual_x._p3_inventory_integrity_catalog import (
    QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS,
)
from tests.qualification.qual_x._p5_freeze_selection_catalog import (
    QUAL_X_P5_FREEZE_SELECTION_MANIFEST,
)
from tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog import (
    QUAL_X_FINAL_CLOSED_WORLD_INVENTORY,
    QUAL_X_FINAL_OWNERSHIP_CORRECTION_START_HEAD,
    QualXFinalInvariantDisposition,
    qual_x_final_catalog_claims_complete_platform_coverage,
    qual_x_final_missing_p3_inventory_owner_ids,
    qual_x_final_provenance_violations,
    qual_x_final_reg01_platform_rollup_is_blocked,
    qual_x_final_unclassified_inventory_owner_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_qual_x_final_ownership_correction_start_head_documented() -> None:
    assert (
        QUAL_X_FINAL_OWNERSHIP_CORRECTION_START_HEAD
        == "cc8c51edb3ae9aa260ae5e87a71dba0527233480"
    )


def test_qual_x_final_closed_world_row_ids_unique() -> None:
    row_ids = [row.row_id for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY]
    assert len(row_ids) == len(set(row_ids))


def test_qual_x_final_p3_inventory_owner_bidirectional_coverage() -> None:
    assert qual_x_final_missing_p3_inventory_owner_ids() == ()
    assert qual_x_final_unclassified_inventory_owner_ids() == ()
    owner_ids = {owner.owner_id for owner in QUAL_X_P3_IN_SCOPE_INVENTORY_OWNERS}
    covered = {
        row.inventory_owner_id
        for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY
        if row.inventory_owner_id is not None
    }
    assert owner_ids <= covered


def test_qual_x_final_evidence_provenance_consistent() -> None:
    assert qual_x_final_provenance_violations() == ()


def test_qual_x_final_reg01_platform_rollup_explicitly_blocked() -> None:
    assert qual_x_final_reg01_platform_rollup_is_blocked()
    assert not qual_x_final_catalog_claims_complete_platform_coverage()


def test_qual_x_final_does_not_equate_catalog_with_full_freeze_checklist() -> None:
    manifest_frz = {
        criterion
        for entry in QUAL_X_P5_FREEZE_SELECTION_MANIFEST
        for criterion in entry.frz_criteria
    }
    catalog_frz = {row.frz_criterion for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY}
    assert "FRZ-REG-01" in manifest_frz
    rollup = next(
        row for row in QUAL_X_FINAL_CLOSED_WORLD_INVENTORY if row.row_id == "CW-PLATFORM-FRZ-ROLLUP"
    )
    assert rollup.disposition is QualXFinalInvariantDisposition.REG_01_PLATFORM_BLOCKED


def test_qual_x_final_synthetic_unclassified_frozen_invariant_owner_would_fail() -> None:
    from tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog import (
        QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
        QualXFinalClosedWorldRow,
        QualXFinalEvidenceProvenance,
        QualXFinalInvariantDisposition,
        qual_x_final_unclassified_inventory_owner_ids,
    )

    broken = (
        QualXFinalClosedWorldRow(
            row_id="SYNTH-UNCLASSIFIED-FROZEN",
            inventory_owner_id="SYNTH-NOT-IN-P3-SCAN-BOUNDARY",
            invariant_summary="synthetic unclassified frozen invariant row",
            stage="SYNTH",
            semantic_owner="SYNTH",
            contract_or_ssot="x",
            production_enforcement="x",
            test_gate="x",
            evidence_sha=QUAL_X_FINAL_DIMENSION_A_EXECUTION_SHA,
            evidence_provenance=QualXFinalEvidenceProvenance.REUSED_HISTORICAL_PYTEST_EXECUTION,
            qualification_gate_introduced_sha=None,
            frz_criterion="FRZ-REG-01",
            disposition=QualXFinalInvariantDisposition.VALID_REUSED_EVIDENCE,
        ),
    )
    saved = QUAL_X_FINAL_CLOSED_WORLD_INVENTORY
    import tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog as catalog_mod

    catalog_mod.QUAL_X_FINAL_CLOSED_WORLD_INVENTORY = saved + broken
    try:
        assert qual_x_final_unclassified_inventory_owner_ids() == (
            "SYNTH-NOT-IN-P3-SCAN-BOUNDARY",
        )
    finally:
        catalog_mod.QUAL_X_FINAL_CLOSED_WORLD_INVENTORY = saved


def test_qual_x_final_synthetic_current_proof_without_execution_sha_would_fail() -> None:
    from tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog import (
        QualXFinalClosedWorldRow,
        QualXFinalEvidenceProvenance,
        qual_x_final_provenance_violations,
    )

    broken = (
        QualXFinalClosedWorldRow(
            row_id="SYNTH-BROKEN",
            inventory_owner_id=None,
            invariant_summary="x",
            stage="SYNTH",
            semantic_owner="SYNTH",
            contract_or_ssot="x",
            production_enforcement="x",
            test_gate="x",
            evidence_sha="",
            evidence_provenance=QualXFinalEvidenceProvenance.EVIDENCE_REQUIRED,
            qualification_gate_introduced_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            frz_criterion="FRZ-REG-02",
            disposition=QualXFinalInvariantDisposition.CURRENT_EXECUTABLE_PROOF,
        ),
    )
    saved = QUAL_X_FINAL_CLOSED_WORLD_INVENTORY
    import tests.qualification.qual_x._qual_x_final_invariant_coverage_catalog as catalog_mod

    catalog_mod.QUAL_X_FINAL_CLOSED_WORLD_INVENTORY = saved + broken
    try:
        assert qual_x_final_provenance_violations()
    finally:
        catalog_mod.QUAL_X_FINAL_CLOSED_WORLD_INVENTORY = saved
